from distutils.core import setup, Extension
from setuptools import find_packages
import sysconfig

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-std=c++11", "-Wall", "-Wextra", "-O3", "-fPIC", "-ffast-math"]
extra_compile_args = [e for e in extra_compile_args if e not in ['-O2']]

trainer = Extension('train', sources = ['sompy3/train.cc'], language='c++11', extra_compile_args=extra_compile_args)
# '/usr/local/lib'

setup(
    name="SOMPY3",
    version="0.9",
    description="Self Organizing Maps Minimal Working Package",
    author="Andreas Loos",
    packages=find_packages(),
    install_requires=['numpy >= 1.7', 'scipy >= 0.9', 'scikit-learn >= 0.16', 'numexpr >= 2.5'],
    ext_modules = [trainer]
)
