# Temporal Saliency Analysis for Multi-Horizon Time Series Forecasting using Deep Learning

Interpreting the model's behavior is important in understanding decision-making in practice. However, explaining complex time series forecasting models faces challenges due to temporal dependencies between subsequent time steps and the varying importance of input features over time. Many time series forecasting models use input context with a look-back window for better prediction performance. However, the existing studies (1) do not consider the temporal dependencies among the feature vectors in the input window and (2) separately consider the time dimension that the feature dimension when calculating the importance scores. In this work, we propose a novel **Windowed Temporal Saliency Analysis** method to address these issues. 

## Saliency Analysis

Saliency Analysis is the study of input feature importance to model output using black-box interpretation techniques. We use the following libraries to perform the saliency analysis methods.

### [Captum](https://captum.ai/docs/introduction)
(“comprehension” in Latin) is an open source library for model interpretability built on PyTorch.

### [Time Interpret (tint)](https://josephenguehard.github.io/time_interpret/build/html/index.html)

This package expands the Captum library with a specific focus on time-series. As such, it includes various interpretability methods specifically designed to handle time series data.

## Multi-Horizon Forecasting
Multi-horizon forecasting is the prediction of variables-of-interest at multiple future time steps. It is a crucial challenge in time series machine learning. Most real-world datasets have a time component, and forecasting the future can unlock great value. For example, retailers can use future sales to optimize their supply chain and promotions, investment managers are interested in forecasting the future prices of financial assets to maximize their performance, and healthcare institutions can use the number of future patient admissions to have sufficient personnel and equipment. 

We use the following library for implementing the time series models,

### [Time-Series-Library (TSlib)](https://github.com/thuml/Time-Series-Library)

TSlib is an open-source library for deep learning researchers, especially deep time series analysis.

## How to Reproduce

The module was developed using python 3.10.

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
## Datasets

The datasets are available at this [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) in the long-term-forecast folder. Download and keep them in the `dataset` folder here. Only `mimic-iii` dataset is private and hence must be approved to get access from [PhysioNet](https://mimic.mit.edu/docs/gettingstarted/).

### Electricity

The electricity dataset [^1] was collected in 15-minute intervals from 2011 to 2014. We select the records from 2012 to 2014 since many
zero values exist in 2011. The processed dataset contains
the hourly electricity consumption of 321 clients. We use
’MT 321’ as the target, and the train/val/test is 12/2/2 months. We aggregated it to 1h intervals following prior works.  

### Traffic

This dataset [^2] records the road occupancy rates from different sensors on San Francisco freeways.

### Mimic-III

MIMIC-III is a private dataset. Refer to [the official MIMIC-III documentation](https://mimic.mit.edu/iii/gettingstarted/dbsetup/). ReadMe and datagen of MIMIC is from [Dynamask Repo](https://github.com/JonathanCrabbe/Dynamask). This repository followed the database setup instructions from [the offficial site here](https://mimic.mit.edu/docs/gettingstarted/local/install-mimic-locally-windows/).

- Run this command to acquire the data and store it:
   ```shell
   python -m data.mimic_iii.icu_mortality --sqluser YOUR_USER --sqlpass YOUR_PASSWORD
   ```
  If everything happens properly, two files named ``adult_icu_vital.gz`` and `adult_icu_lab.gz`
  are stored in `dataset/mimic_iii`.

- Run this command to preprocess the data:
   ```shell
   python -m data.mimic_iii.data_preprocess
   ```
  If everything happens properly, a file `patient_vital_preprocessed.pkl` is stored in `dataset/mimic_iii`.


## Models 
This repository currently has the following models collected from [Time-Series-Library](https://github.com/thuml/Time-Series-Library).

- [x] **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py)
- [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py)
- [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py)
- [x] **MICN** - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=zt53IDUR1U)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MICN.py)
- [x] **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py)
- [x] **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]](https://arxiv.org/abs/2207.01186) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py)
- [x] **ETSformer** - ETSformer: Exponential Smoothing Transformers for Time-series Forecasting [[arXiv 2022]](https://arxiv.org/abs/2202.01381) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py)
- [x] **Non-stationary Transformer** - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/pdf?id=ucNDIDRNjjv) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py)
- [x] **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py)
- [x] **Pyraformer** - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [[ICLR 2022]](https://openreview.net/pdf?id=0EXmFzUn5I) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py)
- [x] **FiLM** - FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/forum?id=zTQdHSQUQWc)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FiLM.py)
- [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py)
- [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py)
- [x] **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py)
- [x] **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py)


## References
<!-- https://docs.github.com/en/enterprise-cloud@latest/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#footnotes -->

[^1]: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014.

[^2]: https://pems.dot.ca.gov/.