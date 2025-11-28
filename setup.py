from setuptools import setup, find_packages

setup(
    name="bioscience-engine",
    version="0.2.0",
    author="Elijah Manda",
    description="Pure Python computational microscopy engine with OpenCV",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "pandas>=1.2.0",
        "scipy>=1.6.0",
        "pillow>=8.0.0",
        "tifffile>=2021.0.0",
        "scikit-image>=0.18.0",
        "matplotlib>=3.3.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-benchmark>=3.4.0",
            "black>=21.0",
            "flake8>=3.9.0"
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
