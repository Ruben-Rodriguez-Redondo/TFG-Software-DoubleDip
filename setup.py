# todo complete
import subprocess
import sys
from setuptools import setup, find_packages

def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()

def install_torch_with_cuda():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cu126",
                          "torch", "torchvision", "torchaudio"])
def install_numpy():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==2.2.3"])

install_torch_with_cuda()
install_numpy() # Install numpy here to avoid errors with torch (in requirements gets duplicated versions)

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
    python_requires=">=3.10, <3.11",
    install_requires=get_requirements(),
    )