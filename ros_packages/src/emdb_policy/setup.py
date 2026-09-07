import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'emdb_policy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test', 'tests']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            os.path.join('share', package_name),
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Fabian Alvarez',
    maintainer_email='alvarez.fabian@outlook.com',
    description='Policy / RL agent node for the EMDB simulator',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_node = emdb_policy.policy_node:main',
            'fruit_shop_bridge = emdb_policy.fruit_shop_bridge:main',
            'ripe_fruit_bridge = emdb_policy.ripe_fruit_bridge:main',
            'train_sb3 = emdb_policy.train_sb3:main',
            'replay_demo = emdb_policy.replay_demo:main',
            'prepare_lift_demo_episodes = emdb_policy.scripts.prepare_lift_demo_episodes:main',
            'publish_demo_episodes = emdb_policy.publish_demo_episodes:main',
        ],
    },
)
