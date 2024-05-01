import torch
import torch.nn as nn
from gymnasium import spaces
import timm

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomModel(nn.Module):
    # def __init__(self, name):
    #     super().__init__()
    #     # We assume CxHxW images (channels first)
    #     # Re-ordering will be done by pre-preprocessing or wrapper
    #     n_input_channels = 6
    #     self.cnn = nn.Sequential(
    #         nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
    #         nn.ReLU(),
    #         nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
    #         nn.ReLU(),
    #         nn.Flatten(),
    #     )

    #     # Compute shape by doing one forward pass
    #     with torch.no_grad():
    #         x = torch.rand(1, 6, 256, 256)
    #         n_flatten= self.cnn(x).shape[1]

    #     self.linear = nn.Sequential(nn.Linear(n_flatten, 256), nn.ReLU())

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.linear(self.cnn(x))
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True
        )
        self.model = self.model.eval()
    def forward(self, x):
        # B, C, H, W = x.shape
        # x = x.reshape(2 * B, C // 2, H, W)
        x = self.model.forward_features(x)
        x = self.model.forward_head(x)
        return x
        # return x.reshape(B, -1)

class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, model_name: str = "swinv2_tiny_window8_256.ms_in1k"):
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        model = CustomModel(model_name)
        with torch.no_grad():
            x = torch.rand(1, 3, 256, 256)
            model_cat_size= model(x).shape
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                extractors[key] = model
                total_concat_size += model_cat_size[1]
            elif key == "EEpos":
                extractors[key] = nn.Linear(subspace.shape[0], 5)
                total_concat_size += 5
            elif key == "Objpos":
                extractors[key] = nn.Linear(subspace.shape[0], 3)
                total_concat_size += 3
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
    
    def forward (self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)
