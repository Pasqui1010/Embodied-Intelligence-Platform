from setuptools import setup
import os
from glob import glob

package_name = 'eip_social_intelligence'

setup(
    name=package_name,
    version='1.0.0',
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
    description='Social Intelligence and Human-Robot Interaction package for the Embodied Intelligence Platform',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'social_intelligence_node = eip_social_intelligence.social_intelligence_node:main',
        ],
    },
) 