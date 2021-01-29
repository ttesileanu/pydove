from distutils.core import setup

setup(
    name="pygrutils",
    version="0.1.5",
    author="Tiberiu Tesileanu",
    author_email="ttesileanu@gmail.com",
    url="https://github.com/ttesileanu/pygrutils",
    packages=["pygrutils"],
    install_requires=[
        "setuptools",
        "statsmodels",
        "matplotlib",
        "seaborn",
        "numpy",
        "scipy",
    ],
)
