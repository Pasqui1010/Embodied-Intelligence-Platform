from setuptools import setup

package_name = 'eip_advanced_learning'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/advanced_learning_demo.launch.py']),
        ('share/' + package_name + '/config', ['config/advanced_learning.yaml']),
        ('share/' + package_name + '/tests', ['tests/test_skill_learning.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='EIP Team',
    maintainer_email='eip@example.com',
    description='Advanced Learning and Adaptation System for Embodied Intelligence Platform',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'learning_engine_node = eip_advanced_learning.learning_engine_node:main'
        ],
    },
)
