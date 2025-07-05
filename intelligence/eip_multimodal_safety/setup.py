from setuptools import setup
import os
from glob import glob

package_name = 'eip_multimodal_safety'

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
    description='Multi-modal safety fusion system for embodied intelligence platform',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'multimodal_safety_node = eip_multimodal_safety.multimodal_safety_node:main',
        ],
    },
) 