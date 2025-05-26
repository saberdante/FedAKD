# FedKAD

This repository is the official PyTorch implementation of:

[**"FedAKD: Federated Adaptive Knowledge Distillation via Global Knowledge Calibration and Decoupling"**]

Thanks for FedNTD. We build this repo based on the [FedNTD (Federated Not-True Distillation)](https://github.com/Lee-Gihun/FedNTD)).

## Requirements

- This codebase is written for `python3` (used `python 3.8.8` while implementing).
- We use Pytorch version of `1.9.0` and `10.2`, `11.0` CUDA version.
- To install necessary python packages,  
    ```
    pip install -r requirements.txt
    ```
- The logs are uploaded to the wandb server. If you do not have a wandb account, just install and use as offline mode. 
  ```
  pip install wandb
  wandb off
  ```

## How to Run Codes?

The configuration skeleton for each algorithm is in `./config/*.json`. 
- `python ./main.py --config_path ./config/algorithm_name.json` conducts the experiment with the default setups.


