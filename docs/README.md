# Sensitivity Analysis for Multi-Horizon Time Series Forecasting

## Sensitivity Analysis

### Definition
According to [Wikipedia](https://en.wikipedia.org/wiki/Sensitivity_analysis)
> Sensitivity analysis is the study of how the uncertainty in the output of a mathematical model or system (numerical or otherwise) can be apportioned to different sources of uncertainty in its inputs.

The sensitivity of each input is often represented by a numeric value, called the sensitivity index. Sensitivity indices come in several forms:

1. **First-order indices**: measures the contribution to the output variance by a single model input alone.
2. **Second-order indices**: measures the contribution to the output variance caused by the interaction of two model inputs.
3. **Total-order index**: measures the contribution to the output variance caused by a model input, including both its first-order effects (the input varying alone) and all higher-order interactions.

## Multi-Horizon Forecasting

###  Definition
Multi-horizon forecasting is the prediction of variables-of-interest at multiple future time steps. It is a crucial challenge in time series machine learning. Most real-world datasets have a time component, and forecasting the future can unlock great value. For example, retailers can use future sales to optimize their supply chain and promotions, investment managers are interested in forecasting the future prices of financial assets to maximize their performance, and healthcare institutions can use the number of future patient admissions to have sufficient personnel and equipment.

The current `Sensitivity Analysis` methods only count for analyzing sensitivity at a single point in time. However in this work we extend that to allow analyzing sensitivity for multiple input (window) and output (horizon) timesteps.

### [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/getting-started.html)

In this work we use `PyTorch Forecasting`  to implement the timeseries models. This framework aims to ease state-of-the-art timeseries forecasting with neural networks for both real-world cases and research alike. The goal is to provide a high-level API with maximum flexibility for professionals and reasonable defaults for beginners. 

## How to Reproduce

### Create Virtual Environment
First create a virtual environment with the required libraries. For example, to create an venv named `ml`, you can either use the `Anaconda` library or your locally installed `python`.

#### Option A: Anaconda
If you have `Anaconda` installed locally, follow the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). An example code,

```
conda create -n ml python=3.10
conda activate ml
```
This will activate the venv `ml`.


#### Option B: Python

If you only have `python` installed but no `pip`, installed pip and activate a virtual env using the following commands from [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/),

On linux/macOS :

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv ml
source ml/bin/activate
python3 -m pip install -r requirements.txt
```

On windows :
```bash
py -m pip install --upgrade pip
py -m pip install --user virtualenv
py -m venv ml
.\env\Scripts\activate
py -m pip install -r requirements.txt
```

Follow the instructions [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) to make sure you have the `pip` and `virtualenv` installed and working. Then create a virtual environement (e.g. name ml) or install required libraries in the default env, using the 

### Install Libraries
Once you have the virtual environment created and running, you can download the libraries using, the [requirement.txt](/requirements.txt) file. 

On linux/macOS :

```bash
python3 -m pip install -r requirements.txt
```

On windows :
```bash
py -m pip install -r requirements.txt
```

You can test whether the environment has been installed properly using a small dataset in the [`train.py`](/TFT-pytorch/script/train_simple.py) file.

### Installing CUDA
The default versions installed with `pytorch-forecasting` might not work and print cpu instead for the following code. Since it doesn't install CUDA with pytorch.

```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} backend')
```

In such case, replace existing CUDA with the folowing version. Anything newer didn't work for now.
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```
