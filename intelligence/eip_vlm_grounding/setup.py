from setuptools import setup
import os
from glob import glob

package_name = 'eip_vlm_grounding'

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
    maintainer='Embodied Intelligence Platform Team',
    maintainer_email='maintainer@embodied-intelligence.org',
    description='Vision-Language Grounding package for spatial reference resolution and object affordance estimation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_grounding_node = eip_vlm_grounding.vlm_grounding_node:main',
        ],
    },
) 