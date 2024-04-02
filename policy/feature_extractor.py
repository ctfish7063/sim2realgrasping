import torch
import torch.nn as nn
from gymnasium import spaces
import timm

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True
        )
        self.model = self.model.eval()
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(2 * B, C // 2, H, W)
        x = self.model.forward_features(x)
        x = self.model.forward_head(x)
        return x.reshape(B, -1)

class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, model_name: str = "swinv2_tiny_window8_256.ms_in1k"):
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        model = CustomModel(model_name)
        with torch.no_grad():
            x = torch.rand(1, 6, 256, 256)
            model_cat_size= model(x).shape
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                extractors[key] = model
                total_concat_size += model_cat_size[1]
            elif key == "EEpos":
                extractors[key] = nn.Linear(subspace.shape[0], 7)
                total_concat_size += 7
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
    
    def forward (self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)
