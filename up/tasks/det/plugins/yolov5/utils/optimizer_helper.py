import torch.nn as nn
from up.utils.model.optimizer_helper import BaseOptimizer
from up.utils.general.registry_factory import OPTIMIZER_REGISTRY


__all__ = ['Yolov5Optimizer']


@OPTIMIZER_REGISTRY.register('yolov5')
class Yolov5Optimizer(BaseOptimizer):
    def get_trainable_params(self, cfg_optim):
        weight_decay = cfg_optim['kwargs']['weight_decay']
        trainable_params = [{"params": []}, {"params": []}, {"params": []}, {"params": []}]

        trainable_params_list = []
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                trainable_params[2]["params"].append(v.bias)  # biases
                trainable_params_list.append(k + ".bias")
                trainable_params[2]["weight_decay"] = 0.0
            if isinstance(v, nn.BatchNorm2d):
                trainable_params[0]["params"].append(v.weight)  # no decay
                trainable_params_list.append(k + ".weight")
                trainable_params[0]["weight_decay"] = 0.0
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                trainable_params[1]["params"].append(v.weight)
                trainable_params_list.append(k + ".weight")
                trainable_params[1]["weight_decay"] = weight_decay

        for n, p in self.model.named_parameters():
            if n not in trainable_params_list:
                trainable_params[-1]["params"].append(p)

        return trainable_params
