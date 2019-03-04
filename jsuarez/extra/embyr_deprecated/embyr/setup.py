#!/usr/bin/env python

from setuptools import find_packages, setup
from os.path import join, dirname
from os import walk

examples = {}
for root, subFolders, files in walk('examples'):
    for fn in files:
        ext = fn.split('.')[-1].lower()
        filename = join(root, fn)
        directory = '%s%s' % ('share/kivy3-', dirname(filename))
        if directory not in examples:
            examples[directory] = []
        examples[directory].append(filename)

setup(
    name='kivy3',
    version='0.1',
    description='Kivy extensions for 3D graphics',
    author='Niko Skrypnik',
    author_email='nskrypnik@gmail.com',
    include_package_data=True,
    packages=find_packages(exclude=("tests",)),
    data_files=list(examples.items()),
    requires=['kivy', ]
)
