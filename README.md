# Extended and Improved Implementation of Double-DIP  

This project is an extended and refined fork of the official implementation of the paper [*"Double-DIP: Unsupervised Image Decomposition via Coupled Deep-Image-Priors"*](http://www.wisdom.weizmann.ac.il/~vision/DoubleDIP/resources/DoubleDIP.pdf),  
originally available [here](https://github.com/yossigandelsman/DoubleDIP/tree/master). Which allow performs tasks as
image segmentation, watermark removal, transparency separation and dehazing, all of them using **unsupervised techniques**.

## 🚀 Enhancements in this version  
- ✅ **Code cleanup and refactoring** for better readability and maintainability.  
- ✅ **Extensive comments** to enhance understanding.  
- ✅ **A user-friendly Gradio interface** for easier interaction.  
- ✅ **Additional functionality**:  
  -  Including hint creation in **Image segmentation**
  -  **Dehazing** extended to videos  

## Table of Contents
- [Installation](#installation) 🛠️
- [Usage](#usage) 💡
- [License](#license) 📜

## Installation 🛠️
Follow these steps to set up the project:  

### 1️⃣ Install Python 3.10  
Make sure you have **Python 3.10** installed (probably higher versions will work to). Can be downloaded from the official site:  
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
**It's necessary to have a GPU compatible with cuda version 12.6 and use Linux or Windows as OS**.
More details [here](https://pytorch.org/). \
Create a virtual environment, activated (consult the IDE or shell guide) and execute:
```
pip install . 
```

## Usage 💡
The code is composed for two main modules, the first one, double_dip_core mainly based in
the [original implementation](https://github.com/yossigandelsman/DoubleDIP/tree/master). From which its posible to
execute the .py of the different functionalities. Each one comes with its respective main to show the usage f.e. for execute de predetermine main
for dehazing images run:
```
python double_dip_core/dehazing.py
```

The second one is double_dip_gradio, which includes the gradio interface and  the wrappers to adapt the originals functionalities implementations
with the gradio interface. To access the interface run: 
```
python double_dip_gradio/app.py 
```
And navigate to http://localhost:7860/, port number can be modified in double_dip_gradio/common/config.json.

# 📸 Gradio interface
<div align="center">
  <img  src = "/figs/main_functionalities.png" alt = "Main Functionalities">
</div>

## License 📜

This work is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
You are free to copy, modify, distribute, and reproduce the material in any medium, provided that you give appropriate credit, indicate if changes were made, and distribute your contributions under the same license.

<div align="center">
  <img src="/figs/license.png" alt="License">
</div>