from setuptools import setup, find_packages
from glob import glob

package_name = 'choirbot_examples'
scripts = {
    'containment':      ['guidance', 'integrator', 'rviz'],
    'bearingformationcontrol': ['guidance', 'integrator' ],
    'doubleintegrator_ca': ['guidance', 'plotter','integrator', 'collision', 'closest_robot_getter'],
    'quadrotorbearingformationcontrol': ['guidance', 'controller', 'integrator' ],
    'formationcontrol': ['guidance', 'controller', 'collision', 'closest_robot_getter', 'plotter'],
    'mpc':              ['guidance', 'integrator', 'rviz'],
    'taskassignment':   ['guidance', 'table', 'planner', 'controller'],
    'webots':   ['closest_robot_getter','guidance', 'controller', 'collision'],
    'FALKO': ['keypoint_detector', 'keypoint_matcher', 'plotter'],
    'collabSLAM':   ['simple_guidance','guidance', 'controller'],
    'FALKOcollabSLAM': ['simple_guidance', 'guidance', 'controller'],
    }

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*.launch.py')),
        ('share/' + package_name, glob('resource/*.rviz')),
        ('share/' + package_name, glob('resource/*.sdf')),
        ('share/' + package_name, glob('resource/*.urdf')),
        ('share/' + package_name + '/worlds', glob('worlds/*.wbt')),

    ],
    install_requires=['setuptools', 'choirbot'],
    zip_safe=True,
    maintainer='OPT4SMART',
    maintainer_email='info@opt4smart.eu',
    description='Example files for ChoiRbot',
    license='GNU General Public License v3.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'choirbot_{0}_{1} = choirbot_examples.{0}.{1}:main'.format(package, file)
            for package, files in scripts.items() for file in files
        ] + [
            'plotter = choirbot_examples.plotter:main'
        ],
    },
)
