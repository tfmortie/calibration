"""
Setup file for C++ NNet module.

Author: Thomas Mortier
Date: March 2022
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules = [
    CppExtension(
        'nnet_cpp',
        ['nnet_cpp.cpp'],
        extra_compile_args=['-O3', '-g', '-fopenmp'],
        extra_link_args=['-lgomp']) # use this line on linux
]
setup(name='nnet_cpp', ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension})
