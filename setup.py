from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os

class BuildExt(build_ext):
    def build_extensions(self):
        # Compile for current platform
        if self.compiler.compiler_type == 'unix':
            for ext in self.extensions:
                ext.extra_compile_args += ['-O3', '-fPIC']
        super().build_extensions()

grpo_module = Extension(
    'grpoptim.c_src.libgrpo',
    sources=['grpoptim/c_src/grpo.c'],
    libraries=['m'],
    extra_compile_args=['-O3', '-fPIC']
)

setup(
    name="grpoptim",
    version="0.1.0",
    author="Subashanan Nair",
    author_email="valiban12@gmil.com",
    description="Group Relative Policy Optimization for Efficient RL Training",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/subaashnair/grpoptim",
    packages=find_packages(),
    ext_modules=[grpo_module],
    install_requires=[
        "numpy>=1.23.0",
        # "ctypes>=1.1.0"
        'torch>=2.0.0'
    ],
    extras_require={
        'test': ['pytest>=7.0', 'torch>=1.10']
    },
    include_package_data=True,
    package_data={
        "grpoptim": ["c_src/*.dylib", "c_src/*.so"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.7',
    cmdclass={'build_ext': BuildExt},
)