from setuptools import setup
import os
from glob import glob

package_name = 'eip_cognitive_architecture'

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
    maintainer='AI Team',
    maintainer_email='ai-team@embodied-intelligence.com',
    description='Cognitive Architecture for Embodied Intelligence Platform',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cognitive_architecture_node = eip_cognitive_architecture.cognitive_architecture_node:main',
        ],
    },
) 