import os
from setuptools import setup, find_packages

setup(
    name="floro",
    version="0.1.0",
    author="Jorge L. Rodriguez",
    author_email="jorlrodriguezg@gmail.com",
    description="FLORO: A Multimodal Geospatial Foundation Model for Ecological Remote Sensing Across Sensors and Scales",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/jorlrodriguezg/floro",
    packages=find_packages(where="src"),
    package_dir={"" : "src"},
    python_requires=">=3.8",
    install_requires=[
        "einops",
        "rasterio",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)