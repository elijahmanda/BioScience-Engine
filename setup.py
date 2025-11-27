from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "bioscience_engine.core.bindings",
        sources=[
            "bioscience_engine/core/bindings.pyx",
            "bioscience_engine/core/fast_ops.cpp",
        ],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-march=native"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="bioscience-engine",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
    }),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.8",
)