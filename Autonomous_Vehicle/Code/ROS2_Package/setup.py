from setuptools import setup
import os
from glob import glob

package_name = 'control_stack'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'data/control'), glob('data/control/*')),
        (os.path.join('share', package_name, 'data/path_planner'), glob('data/path_planner/*')),
        (os.path.join('share', package_name, 'data/object_detection'), glob('data/object_detection/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Asher Elmquist',
    maintainer_email='amelmquist@wisc.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_path_follower_final_project = control_stack.control_path_follower_final_project:main',
            'simulation_final_project = control_stack.simulation_final_project:main',
            'simulation = control_stack.simulation:main',
            'control_open_loop = control_stack.control_open_loop:main',
            'control_path_follower = control_stack.control_path_follower:main',
            'path_planner_from_file = control_stack.path_planner_from_file:main',
            'object_detection_from_file = control_stack.object_detection_from_file:main',
            'path_planner = control_stack.path_planner:main',
            'localizer = control_stack.localizer:main',
            'object_recognition = control_stack.object_recognition:main'
        ],
    },
)
