from setuptools import setup
from os.path import join, dirname, abspath

import re

re_ver = re.compile(r"__version__\s+=\s+'(.*)'")
with open(join(abspath(dirname(__file__)), 'nntime', '__init__.py'), encoding='utf-8') as f:
    version = re_ver.search(f.read()).group(1)


setup(
    name='nntime',
    version=version,
    description='Timing utilities for deep learning modules in PyTorch',
    long_description='See project page: https://github.com/mtli/nntime',
    url='https://github.com/mtli/nntime',
    author='Mengtian (Martin) Li',
    author_email='martinli.work@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='deep learning pytorch timing profiling',
    packages=['nntime'],
    python_requires='>=3',
    install_requires=[
        'numpy',
    ],
    include_package_data = True,
)
