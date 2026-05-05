"""Setup script for the maze_bringup package (launch files + YAML configs)."""

import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'maze_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config', 'mazes'),
            glob('config/mazes/*.yaml')),
        (os.path.join('share', package_name, 'config', 'markers'),
            glob('config/markers/*.yaml')),
        (os.path.join('share', package_name, 'config', 'sweeps'),
            glob('config/sweeps/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Salvador Santos',
    maintainer_email='salvadorvelososantos@gmail.com',
    description='Launch files and configuration for the AlphaBot2 maze-solving stack.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
