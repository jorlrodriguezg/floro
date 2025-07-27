from setuptools import setup, find_packages

setup(
    name="floro",
    version="0.1.0",
    author="Jorge L. Rodriguez",
    author_email="jorlrodriguezg@gmail.com",
    description="FLORO: Fusion Learning Of Remote Sensing Observations for Ecological Research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jorlrodriguezg/floro",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7"    
)
