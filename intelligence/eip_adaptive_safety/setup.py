from setuptools import setup
import os
from glob import glob

package_name = 'eip_adaptive_safety'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='EIP Development Team',
    maintainer_email='dev@embodied-intelligence.com',
    description='Adaptive Safety Orchestration (ASO) for Embodied Intelligence Platform',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'adaptive_safety_node = eip_adaptive_safety.adaptive_safety_node:main',
        ],
    },
) 