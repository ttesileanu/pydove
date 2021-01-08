from distutils.core import setup

setup(
    name="pygrutils",
    version="0.1.1",
    author="Tiberiu Tesileanu",
    author_email="ttesileanu@gmail.com",
    url="https://github.com/ttesileanu/pygrutils",
    packages=["pystanic"],
    install_requires=[
        "setuptools",
        "statsmodels",
        "matplotlib",
    ],
)
