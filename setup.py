"""The setup script."""

import setuptools
from regression_analysis import __version__

__author__ = "Amandine Kaiser, Anthony Val Camposano, Harish Pruthviraj Jain"
__contact__ = "amandine.kaiser@geo.uio.no"
__license__ = "Apache licence"

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name="Regression Analysis and Resampling Methods",
    version=__version__,
    author=__author__,
    author_email=__contact__,
    license=__license__,
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/am-kaiser/CompSci-Project-1.git",
    packages=setuptools.find_packages(include=["regression_analysis", "regression_analysis.*"]),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Operating System :: Microsoft :: Windows',
        'Natural Language :: English',
    ],
    python_requires='>=3.6',
)