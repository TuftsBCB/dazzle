# DAZZLE

This repository include code and documentation for our manuscript "Improving Gene Regulatory Network Inference using Dropout Augmentation". 

Checkout RegDiffusion, our latest work on gene regulatory network inference: https://github.com/TuftsBCB/RegDiffusion

## Install

This package is available on pip 

```
pip install grn-dazzle
```

## Basic Usage

The core function `runDAZZLE` requires the following two things to get started:

- **Single cell gene expression table**. We suggest you use log transformation to normalize the data
- **Experiment Configs**. We also provide two sets of default configs with this package, namely `DEFAULT_DAZZLE_CONFIGS` and `DEFAULT_DEEPSEM_CONFIGS`. They are just two python dictionaries. If you need to make modifications, just save them to a variable and adjust the values. 

## Quick Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kwRG0dsqJAHxsOXF9zFeyNxpuh_TWSGg?usp=sharing)

```
from dazzle import load_beeline, runDAZZLE, get_metrics, DEFAULT_DAZZLE_CONFIGS


bl_data, bl_ground_truth = load_beeline(
    data_dir='data', 
    benchmark_data="hESC", 
    benchmark_setting="500_STRING"
)

model, adjs = runDAZZLE(bl_data.X, DEFAULT_DAZZLE_CONFIGS)

get_metrics(model.get_adj(), bl_ground_truth)
```
