from setuptools import find_packages, setup

package_name = 'maze_mdp'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Salvador Santos',
    maintainer_email='salvadorvelososantos@gmail.com',
    description=(
        'MDP, Value Iteration, SARSA and Q-Learning for autonomous maze '
        'solving, plus a Python micro-simulator.'
    ),
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
        'analysis': [
            'pandas>=1.5',
            'matplotlib>=3.5',
        ],
    },
    entry_points={
        'console_scripts': [
            'maze_publisher = maze_mdp.nodes.maze_publisher:main',
            'fiducial_localizer = maze_mdp.nodes.fiducial_localizer:main',
            'policy_runner = maze_mdp.nodes.policy_runner:main',
            'action_executor = maze_mdp.nodes.action_executor:main',
            'ir_driver_sim = maze_mdp.nodes.ir_driver_sim:main',
            'cell_tracker = maze_mdp.nodes.cell_tracker:main',
            'maze_sim_node = maze_mdp.nodes.maze_sim_node:main',
            'maze_viz_node = maze_mdp.nodes.maze_viz_node:main',
            'compare_node = maze_mdp.nodes.compare_node:main',
            'train = maze_mdp.experiments.runner:main',
            'sweep = maze_mdp.experiments.sweep:main',
        ],
    },
)
