#!/usr/bin/env python

from distutils.core import setup

setup(name='mdpcl',
      version='0.6',
      description='decorators to compile Python code to C99, OpenCL, and JS',
      author='Massimo Di Pierro',
      author_email='massimo.dipierro@gmail.com',
      license='bsd',
      url='https://github.com/mdipierro/mdpcl',
      scripts = ['mdpcl.py'],
      py_modules = ['meta'],
      )

