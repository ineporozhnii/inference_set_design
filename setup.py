from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="inference_set_design",
    version="0.1",
    packages=find_packages(),
    install_requires=required,
)
