from setuptools import find_packages, setup

package_name = 'pack'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'opencv-python',
        'ultralytics',
        'numpy',
        'scikit-learn',
        'cv_bridge',
        'imutils',
        'PyYAML'
    ],
    zip_safe=True,
    maintainer='pritha',
    maintainer_email='cs23b1099@iiitdm.ac.in',
    description='TODO: Package description',
    license='TODO: License declaration',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node = pack.node:main',
            'cvnode=pack.cv:main',
            'cvnode2=pack.cv2:main',
            'node2=pack.node2:main',
            'node3=pack.node3:main',
            'cvnode3=pack.cv3:main',
            'node4=pack.node4:main',
            'node5=pack.node5:main'
        ],
    },
)
