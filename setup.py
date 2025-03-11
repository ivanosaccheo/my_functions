from setuptools import setup, find_packages

setup(
    name="my_library",  # Name of your package
    version="0.1",
    packages=find_packages(),  # Automatically find submodules
    install_requires=[],  # List dependencies here if needed
)
