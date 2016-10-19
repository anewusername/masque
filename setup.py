#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='masque',
      version='0.1',
      description='Lithography mask library',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/gogs/jan/masque',
      packages=find_packages(),
      install_requires=[
            'numpy'
      ],
      extras_require={
          'visualization': ['matplotlib'],
          'gdsii': ['python-gdsii'],
          'svg': ['svgwrite'],
          'text': ['freetype-py', 'matplotlib']
      },
      )

