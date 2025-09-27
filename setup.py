from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import os
import subprocess

def get_eigen_include():
    try:
        result = subprocess.run(['pkg-config', '--cflags-only-I', 'eigen3'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().replace('-I', '')
    except:
        pass
    
    # Fallback paths
    eigen_paths = [
        '/usr/include/eigen3',
        '/usr/local/include/eigen3',
        '/opt/homebrew/include/eigen3',
        'bioscience_engine/core/cpp/include/eigen'
    ]
    
    for path in eigen_paths:
        if os.path.exists(path):
            return path
    
    raise RuntimeError("Eigen3 not found. Please install eigen3-dev")

extensions = [
    Extension(
        "bioscience_engine.core.bindings",
        sources=[
            "bioscience_engine/core/bindings.pyx",
            "bioscience_engine/core/cpp/src/engine.cpp",
            "bioscience_engine/core/cpp/src/image_processor.cpp",
            "bioscience_engine/core/cpp/src/cell_detector.cpp",
            "bioscience_engine/core/cpp/src/tracker.cpp",
            "bioscience_engine/core/cpp/src/memory_pool.cpp",
        ],
        include_dirs=[
            np.get_include(),
            "bioscience_engine/core/cpp/include",
            get_eigen_include(),
        ],
        language="c++",
        extra_compile_args=[
            "-std=c++17",
            "-O3",
            "-fopenmp",
            "-march=native",
            "-DEIGEN_USE_MKL_ALL" if os.environ.get('USE_MKL') else "",
        ],
        extra_link_args=["-fopenmp"],
        libraries=["tiff"],
    )
]

setup(
    name="bioscience-engine",
    version="0.1.0",
    description="High-Performance Computational Microscopy Engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Elijah Manda",
    author_email="elijahmandajc@gmail.com",
    url="https://github.com/elijahmanda/bioscience-engine",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives={
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'language_level': 3,
    }),
    install_requires=open("requirements.txt").read().splitlines(),
    extras_require={
        'dev': open("requirements-dev.txt").read().splitlines(),
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    zip_safe=False,
)
