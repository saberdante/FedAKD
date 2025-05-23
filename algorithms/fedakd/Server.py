import torch
import torch.nn as nn
import numpy as np
import copy
import time
import os
import sys
import wandb
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseServer import BaseServer
from algorithms.fedakd.ClientTrainer import ClientTrainer
from algorithms.fedakd.criterion import *
from algorithms.measures import *

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler=None, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        local_criterion = self._get_local_criterion(self.algo_params, self.num_classes)

        self.client = ClientTrainer(
            local_criterion,
            algo_params=algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )

        # Count the Major labels 存储每个客户端的主要标签
        self.client_y_lst = {}
        # 记录每个客户端分布的概率-用于后面赋值权重使用
        self.client_data_map_dict = {}
        for client_idx, d in self.data_distributed["local"].items():
            data_map_value = self.data_distributed["data_map"][client_idx] / sum(self.data_distributed["data_map"][client_idx])
            final_data_map_value = data_map_value
            mean_v = np.mean(self.data_distributed["data_map"][client_idx])  
            y_lst = np.where(np.array(self.data_distributed["data_map"][client_idx]) > mean_v)[0] 
            self.client_y_lst[client_idx] = copy.deepcopy(torch.tensor(y_lst, device=self.device)) 
            self.client_data_map_dict[client_idx] = copy.deepcopy(torch.tensor(final_data_map_value, device=self.device))
        print("\n>>> fedakd Server initialized...\n")

    def _get_local_criterion(self, algo_params, num_classes):
        tau = algo_params.tau
        beta = algo_params.beta
        gamma = algo_params.gamma
        criterion = akd_loss(num_classes, tau, beta, gamma)

        return criterion

    def _set_client_data(self, client_idx):
        super()._set_client_data(client_idx)
        self.client.major_labels = self.client_y_lst[client_idx]
        self.client.data_map_value = self.client_data_map_dict[client_idx]
        self.client.data_distribution_map = self.data_distributed["data_map"][client_idx]