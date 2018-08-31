#!/usr/bin/env python

from setuptools import setup, find_packages
import masque

setup(name='masque',
      version=masque.version,
      description='Lithography mask library',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/code/jan/masque',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'pyclipper',
      ],
      extras_require={
          'visualization': ['matplotlib'],
          'gdsii': ['python-gdsii'],
          'svg': ['svgwrite'],
          'text': ['freetype-py', 'matplotlib'],
      },
      )

