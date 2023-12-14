from setuptools import find_packages, setup

package_name = 'aruco_pose'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sophia',
    maintainer_email='sophia@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_pub = aruco_pose.camera_pub:main',
            'camera_sub = aruco_pose.camera_sub:main'
        ],
    },
)
