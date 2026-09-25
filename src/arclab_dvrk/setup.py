from setuptools import setup
import os
from glob import glob

package_name = 'arclab_dvrk'

setup(
    name=package_name,
    version='0.0.1',
    packages=['arclab_dvrk', 'arclab_dvrk.tools', 'arclab_dvrk.data_collection'],
    package_dir={
        'arclab_dvrk': 'src',
        'arclab_dvrk.tools': 'src/tools',
        'arclab_dvrk.data_collection': 'src/data_collection'
    },
    py_modules=[
        'arclab_dvrk.psm_control',
        'arclab_dvrk.utils',
        'arclab_dvrk.tools.cautery_spatula',
        'arclab_dvrl.data_collection.writer'
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Description of your package',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)