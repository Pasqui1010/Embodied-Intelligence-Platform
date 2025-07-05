from setuptools import setup
import os
from glob import glob

package_name = 'eip_safety_simulator'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='EIP Maintainers',
    maintainer_email='maintainers@embodied-intelligence-platform.org',
    description='Digital Twin Safety Ecosystem for embodied intelligence platform',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'safety_simulator_node = eip_safety_simulator.safety_simulator_node:main',
            'scenario_generator = eip_safety_simulator.scenario_generator:main',
            'safety_validator = eip_safety_simulator.safety_validator:main',
        ],
    },
) 