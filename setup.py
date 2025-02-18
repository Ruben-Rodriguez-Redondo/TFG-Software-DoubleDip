# todo complete
from setuptools import setup, find_packages

def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()

setup(
    name="DoubleDIP",
    version="1.0.0",
    packages=find_packages(),
    author_email="r.rodriguezr.2020@alumnos.urjc.es",
    description="DoubleDIP functionalities module with gradio interface",
    url="https://github.com/Ruben-Rodriguez-Redondo/TFG-Software-DoubleDip",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=get_requirements(),
)