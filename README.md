# Extended and Improved Implementation of Double-DIP

This project is an extended and refined fork of the official implementation of the paper [*"Double-DIP: Unsupervised
Image Decomposition via Coupled
Deep-Image-Priors"*](http://www.wisdom.weizmann.ac.il/~vision/DoubleDIP/resources/DoubleDIP.pdf),  
originally available [here](https://github.com/yossigandelsman/DoubleDIP/tree/master). Which allow performs tasks as
image segmentation, watermark removal, transparency separation and dehazing, all of them using **unsupervised techniques
**.

## 🚀 Enhancements in this version

- ✅ **Code cleanup and refactoring** for better readability and maintainability.
- ✅ **Bug fix and extensive comments** to enhance understanding.
- ✅ **A user-friendly Gradio interface** for easier interaction.
- ✅ **GPU, CPU and torch.dtype selection** for more accessibility, personalization and speed.
- ✅ **SSIM metrics** to evaluate the effectiveness of the tasks.
- ✅ **Additional functionality**:
    - Including hint creation and main example in **Image segmentation**
    - **Dehazing** extended to videos (.mp4 format).

## Table of Contents

- [Installation](#installation) 🛠️
- [Usage](#usage) 💡
- [License](#license) 📜

## Installation 🛠️

Follow these steps to set up the project:

### 1️⃣ Install Python 3.10

Make sure you have **Python 3.10** installed (probably higher versions will work to). Can be downloaded from the
official site:  
🔗 [Python Downloads](https://www.python.org/downloads/)

To check your Python version, run in the local terminal:

```
python --version
```

### 2️⃣ Clone the repository

```
git clone https://github.com/Ruben-Rodriguez-Redondo/TFG-Software-DoubleDip 
cd TFG-Software-DoubleDip  
```

### 3️⃣ ️Install dependencies

**If you want to use the GPU it is necessary to have a GPU compatible with cuda versions 11.8-12.6 and use Linux or
Windows as OS**.
More details [here](https://pytorch.org/). \
Create a virtual environment, activated (consult the IDE or shell guide). In Pycharm is enough executing in local
terminal

```
python -m venv .venv
.venv\Scripts\activate
```

Once the environment is created and activated run:

`Instalation with GPU and CPU `

```
python.exe -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install . 
```

This index refer to CUDA version 12.6 but also works with older cuda versions until 11.8 (included), just change the
index-url.

`Instalation with only CPU`

```
python.exe -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install . 
```

More information about the different indexes and its torch versions
compatibilities [here](https://pytorch.org/get-started/previous-versions/). And if the package is going to be modified
run the following command to make the changes effective automatically.

```
pip install -e .
```

`Optional`: At least in my Pycharm community version (2024.2.5) even if the packages are correctly installed, and the
code works fine
some packages are still red-marked in the code as not installed. To avoid this warning (if happened to you) just add
.venv\Lib\Site-Packages to
your interpreter paths. More
details [here](https://stackoverflow.com/questions/31235376/pycharm-doesnt-recognize-installed-module).

## Usage 💡

The code is composed for two main modules, the first one, double_dip_core mainly based in
the [original implementation](https://github.com/yossigandelsman/DoubleDIP/tree/master). From which its posible to
execute the .py of the different functionalities. Each one comes with its respective main to show the usage f.e. for
execute de predetermine main
for dehazing images run:

```
python double_dip_core/dehazing.py
```

The second one is double_dip_gradio, which includes the gradio interface and the wrappers to adapt the originals
functionalities implementations
with the gradio interface. To access the interface run:

```
python double_dip_gradio/app.py 
```

And navigate to http://localhost:7860/, port number can be modified in double_dip_gradio/common/config.json.

# 📸 Gradio interface

<div align="center">
  <img  src = "/figs/main_functionalities.png" alt = "Main Functionalities">
</div>

## Bugs & Issues 💡
- Here's a list of some bugs or issues it may occur:
  - ```INFO: Could not find files for the given pattern(s)```. This is a message which apparently appear when the Gradio app starts in Windows 11. Nowadays the  [issue](https://github.com/gradio-app/gradio/issues/9974) is still open. But it doesn't affect the program, is just annoying.
  - ```[WinError 10054] An existing connection was forcibly closed by the remote host"```. This happened mainly when trying to upload a long video in the Gradio app. It doesn't crash the program, but maybe you have to retry until the video is uploaded correctly. There's not only one reason for it to happen but apparently is caused by some incompatibilities between the browser and your binaries [stackoverflow](https://stackoverflow.com/questions/59633068/connectionreseterror-winerror-10054-an-existing-connection-was-forcibly-close)
  - ```
    ERROR: Exception in ASGI application
    ...
    h11._util.LocalProtocolError: Too little data for declared Content-Length
    ```
    This is similar to the previous one but happens with big images instead of videos. It also doesn't crash the  programs, but you have to retry until the image is uploaded correctly. Some of them solved it [changing Gradio version](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/11855) or upgrading [third party libraries](https://github.com/facebookresearch/audio2photoreal/issues/6)


## License 📜

This work is licensed under
the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
You are free to copy, modify, distribute, and reproduce the material in any medium, provided that you give appropriate
credit, indicate if changes were made, and distribute your contributions under the same license.

<div align="center">
  <img src="/figs/license.png" alt="License">
</div>