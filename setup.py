#!/usr/bin/env python3

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

with open('masque/VERSION', 'r') as f:
    version = f.read().strip()

setup(name='masque',
      version=version,
      description='Lithography mask library',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/code/jan/masque',
      packages=find_packages(),
      package_data={
          'masque': ['VERSION']
      },
      install_requires=[
            'numpy',
      ],
      extras_require={
          'visualization': ['matplotlib'],
          'gdsii': ['python-gdsii'],
          'svg': ['svgwrite'],
          'text': ['freetype-py', 'matplotlib'],
      },
      classifiers=[
            'Programming Language :: Python :: 3',
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Information Technology',
            'Intended Audience :: Manufacturing',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU Affero General Public License v3',
            'Operating System :: POSIX :: Linux',
            'Operating System :: Microsoft :: Windows',
            'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)',
            'Topic :: Scientific/Engineering :: Visualization',
      ],
      )

