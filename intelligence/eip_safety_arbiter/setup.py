from setuptools import setup

package_name = 'eip_safety_arbiter'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='EIP Team',
    maintainer_email='maintainers@embodied-intelligence-platform.org',
    description='Safety verification and behavior arbitration for LLM-guided robotic systems',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'safety_monitor_node = eip_safety_arbiter.safety_monitor:main',
        ],
    },
) 