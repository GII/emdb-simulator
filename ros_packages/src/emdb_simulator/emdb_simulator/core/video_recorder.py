#!/usr/bin/env python3
"""Per-episode offscreen video recording for SceneLoader.

EpisodeRecordingSpec parses a compact "which episodes to record" string
(e.g. "all", "5", "0-2,10-12") and VideoRecorder owns a single video
writer at a time, one file per recorded episode, driven by the node's
episode-reset and per-step hooks.

Frames are encoded with imageio's ffmpeg plugin using libx264 at an
explicit CRF (Constant Rate Factor), the same approach robosuite's own
reference recording script uses (robosuite/demos/demo_video_recording.py)
-- this gives real, controllable quality instead of cv2.VideoWriter's
opaque, low-bitrate mp4v default.
"""
import os
import time

import imageio


class EpisodeRecordingSpec:
    """Parses record_video_episodes strings like 'all', '5', '0-2,10-12'.

    Never raises. Malformed tokens are logged as a warning and skipped.
    If nothing valid remains (empty/blank spec, or every token was bad),
    falls back to matching NO episodes -- a typo must not silently start
    recording every episode and fill disk.
    """

    def __init__(self, match_all=False, ranges=None):
        self._match_all = match_all
        self._ranges = ranges or []

    @classmethod
    def parse(cls, spec, logger=None):
        text = (spec or "").strip()
        if text.lower() in ("all", "*"):
            return cls(match_all=True)

        ranges = []
        for raw_token in text.split(","):
            token = raw_token.strip()
            if not token:
                continue
            try:
                if "-" in token:
                    lo_str, hi_str = token.split("-", 1)
                    lo, hi = int(lo_str.strip()), int(hi_str.strip())
                    if lo > hi:
                        raise ValueError(f"range start {lo} > end {hi}")
                else:
                    lo = hi = int(token)
                ranges.append((lo, hi))
            except ValueError as exc:
                if logger is not None:
                    logger.warning(
                        f"record_video_episodes: ignoring invalid token {token!r} ({exc})"
                    )

        if not ranges and logger is not None and text:
            logger.warning(
                f"record_video_episodes={spec!r} has no valid episode ids/ranges; "
                "no episodes will be recorded."
            )

        return cls(match_all=False, ranges=ranges)

    def contains(self, episode_id):
        if self._match_all:
            return True
        return any(lo <= episode_id <= hi for lo, hi in self._ranges)


class VideoRecorder:
    """Owns one video writer at a time (one file per recorded episode)."""

    def __init__(
        self,
        *,
        enabled,
        output_dir,
        episode_spec,
        camera_name,
        width,
        height,
        fps,
        stride,
        crf,
        keep_successes=False,
        logger,
    ):
        self.enabled = bool(enabled)
        self.output_dir = output_dir
        self.episode_spec = episode_spec
        self.camera_name = camera_name
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps and fps > 0 else 20.0
        self.stride = max(1, int(stride))
        self.crf = int(crf)
        # When True, an episode outside episode_spec is still recorded (so we
        # can tell whether it succeeds) and its file is kept if it does,
        # deleted otherwise. An in-range episode is always kept regardless.
        self.keep_successes = bool(keep_successes)
        self.logger = logger

        self._run_dir = None
        self._writer = None
        self._current_episode_id = None
        self._current_path = None
        self._current_in_range = False
        self._episode_success = False
        self._step_counter = 0
        self._camera_unavailable = False
        # keep_successes-only episodes (not in episode_spec) are "deferred":
        # instead of rendering/encoding every step just to maybe throw it
        # away, buffer cheap sim state and only render+encode after the
        # fact, if the episode actually succeeds. See maybe_start_episode/
        # capture_frame/_finish_deferred_episode.
        self._deferred = False
        self._deferred_states = []
        self._env = None

    def _ensure_run_dir(self):
        if self._run_dir is not None:
            return self._run_dir
        t1, t2 = str(time.time()).split(".")
        self._run_dir = os.path.join(self.output_dir, f"run_{t1}_{t2}")
        os.makedirs(self._run_dir, exist_ok=True)
        return self._run_dir

    def _open_writer(self, episode_id):
        """Returns (writer, path), or (None, None) and sets _camera_unavailable on failure."""
        run_dir = self._ensure_run_dir()
        path = os.path.join(run_dir, f"episode_{int(episode_id):04d}.mp4")

        try:
            writer = imageio.get_writer(
                path,
                fps=self.fps,
                codec="libx264",
                pixelformat="yuv420p",
                quality=None,
                # ultrafast trades away compression efficiency (bigger files
                # at a given CRF) for encode speed -- worth it here since
                # capture_frame() runs synchronously in the sim step loop, so
                # slower presets directly cost simulation throughput, not
                # just wall-clock time on some separate encoding job.
                output_params=["-crf", str(self.crf), "-preset", "ultrafast"],
                macro_block_size=None,
            )
        except Exception as exc:
            if self.logger is not None:
                self.logger.error(
                    f"Could not open video writer for {path} ({exc}); "
                    "recording disabled for this run"
                )
            self._camera_unavailable = True
            return None, None
        return writer, path

    def maybe_start_episode(self, episode_id):
        self.close_episode()  # finalize the previous episode first

        if not self.enabled or self._camera_unavailable:
            return
        in_range = self.episode_spec.contains(episode_id)
        if not in_range and not self.keep_successes:
            self._current_episode_id = None
            return

        self._current_episode_id = episode_id
        self._current_in_range = in_range
        self._episode_success = False
        self._step_counter = 0

        if in_range:
            # Explicitly requested -- record live, kept regardless of outcome.
            self._deferred = False
            self._writer, self._current_path = self._open_writer(episode_id)
            if self._writer is not None and self.logger is not None:
                self.logger.info(f"Recording episode {episode_id} -> {self._current_path}")
        else:
            # Only wanted if this episode succeeds (keep_successes). Buffer
            # cheap sim state per step instead of rendering/encoding live --
            # render+encode happens once, after the fact, only if it
            # succeeds (see close_episode/_finish_deferred_episode). Keeps
            # keep_successes-only training runs at full physics speed.
            self._deferred = True
            self._deferred_states = []

    def capture_frame(self, env, success=False):
        if self._camera_unavailable or self._current_episode_id is None:
            return
        self._env = env

        if success:
            self._episode_success = True

        self._step_counter += 1
        if (self._step_counter - 1) % self.stride != 0:
            return

        if self._deferred:
            self._deferred_states.append(env.sim.get_state())
            return

        if self._writer is None:
            return
        try:
            rgb = env.sim.render(
                width=self.width, height=self.height, camera_name=self.camera_name
            )
            # robosuite/MuJoCo offscreen frames are rendered upside down
            # relative to normal image row order.
            self._writer.append_data(rgb[::-1])
        except Exception as exc:
            if self.logger is not None:
                available = None
                try:
                    available = list(env.sim.model.camera_names)
                except Exception:
                    pass
                self.logger.error(
                    f"Video capture failed for camera={self.camera_name!r} ({exc}); "
                    f"available cameras={available}. Disabling further recording for this run."
                )
            self._camera_unavailable = True
            self.close_episode()

    def mark_success(self):
        """Flag the current episode as successful out-of-band, without a
        matching capture_frame() call -- for callers (e.g.
        scene_loader's /mark_episode_success) whose success signal can
        arrive after the last real physics step of an episode, when no
        further capture_frame(success=True) call is coming to record it."""
        if self._current_episode_id is not None:
            self._episode_success = True

    def _finish_deferred_episode(self):
        """Render+encode a buffered episode now that we know it succeeded.

        Replays each buffered (qpos, qvel) snapshot by writing it straight
        into the live `env.sim` and rendering -- cheaper than re-running
        physics, but it does mean this temporarily clobbers whatever
        current state `env.sim` holds (the *next* episode's fresh
        `env.reset()`, already applied by the time this runs -- see
        scene_loader._reset_episode_cb). Always restored in `finally`.
        """
        states = self._deferred_states
        episode_id = self._current_episode_id
        succeeded = self._episode_success
        env = self._env
        self._deferred_states = []
        self._deferred = False

        if not states or not succeeded or env is None:
            return

        writer, path = self._open_writer(episode_id)
        if writer is None:
            return

        restore_state = env.sim.get_state()
        ok = True
        try:
            for state in states:
                env.sim.set_state(state)
                env.sim.forward()
                rgb = env.sim.render(
                    width=self.width, height=self.height, camera_name=self.camera_name
                )
                writer.append_data(rgb[::-1])
        except Exception as exc:
            ok = False
            if self.logger is not None:
                self.logger.error(
                    f"Deferred video replay failed for episode {episode_id} ({exc}); discarding."
                )
        finally:
            env.sim.set_state(restore_state)
            env.sim.forward()
            writer.close()

        if not ok:
            try:
                os.remove(path)
            except OSError:
                pass
        elif self.logger is not None:
            self.logger.info(f"Episode {episode_id} succeeded -- saved recording -> {path}")

    def close_episode(self):
        if self._deferred:
            self._finish_deferred_episode()
            self._current_episode_id = None
            self._current_in_range = False
            self._episode_success = False
            self._step_counter = 0
            return

        path = self._current_path
        keep = path is not None and self._current_in_range

        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None

        if path is not None and not keep:
            try:
                os.remove(path)
                if self.logger is not None:
                    self.logger.info(
                        f"Episode {self._current_episode_id} outside record_video_episodes "
                        f"and not successful; discarded {path}"
                    )
            except Exception as exc:
                if self.logger is not None:
                    self.logger.warning(f"Could not remove episode video {path}: {exc}")

        self._current_episode_id = None
        self._current_path = None
        self._current_in_range = False
        self._episode_success = False
        self._step_counter = 0

    def close(self):
        self.close_episode()
