from distutils.core import setup

setup(
    name="pydove",
    version="0.2.0",
    author="Tiberiu Tesileanu",
    author_email="ttesileanu@gmail.com",
    url="https://github.com/ttesileanu/pydove",
    packages=["pydove"],
    install_requires=[
        "setuptools",
        "statsmodels",
        "matplotlib",
        "seaborn",
        "numpy",
        "scipy",
    ],
)
