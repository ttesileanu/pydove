import pathlib
from setuptools import setup

README = pathlib.Path("README.md").read_text()

setup(
    name="pydove",
    version="0.3.5",
    description="An assortment of graphics utilities",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Tiberiu Tesileanu",
    author_email="ttesileanu@gmail.com",
    license="MIT",
    url="https://github.com/ttesileanu/pydove",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    packages=["pydove"],
    install_requires=[
        "setuptools",
        "statsmodels",
        "matplotlib>=3.1.1",
        "seaborn",
        "numpy>=1.16.4",
        "scipy",
    ],
    include_package_data=True,
)
