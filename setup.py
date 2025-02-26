from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="neuroloader",
    version="0.1.0",
    author="Kyle Curham",
    author_email="kyle@curham.com",
    description="A package for downloading and processing neuroimaging data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kyle-curham/neuroloader",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
) 