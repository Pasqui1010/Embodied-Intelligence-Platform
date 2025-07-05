from setuptools import setup

package_name = 'eip_slam'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'sensor_msgs_py'],
    zip_safe=True,
    maintainer='EIP Team',
    maintainer_email='maintainers@embodied-intelligence-platform.org',
    description='Basic SLAM implementation for the Embodied Intelligence Platform',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'basic_slam_node = eip_slam.basic_slam_node:main',
        ],
    },
) 