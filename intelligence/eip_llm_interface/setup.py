from setuptools import setup
import os
from glob import glob

package_name = 'eip_llm_interface'

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
    maintainer='EIP Maintainers',
    maintainer_email='maintainers@embodied-intelligence-platform.org',
    description='LLM interface layer for embodied intelligence platform',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_interface_node = eip_llm_interface.llm_interface_node:main',
            'llm_planning_node = eip_llm_interface.llm_planning_node:main',
        ],
    },
) 