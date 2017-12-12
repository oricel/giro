# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='awalker',
    version='0.1.0',
    description='A Random Walker',
    long_description=readme,
    author='OC',
    author_email='',
    url='https://github.com/oricel/giro',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

