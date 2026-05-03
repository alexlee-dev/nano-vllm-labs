import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    params = dict(model.named_parameters())
    loaded: set[str] = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = params.get(param_name)
                        if param is None:
                            break
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, loaded_weight, shard_id)
                        loaded.add(param_name)
                        break
                else:
                    param = params.get(weight_name)
                    if param is None:
                        if (
                            weight_name == "model.embed_tokens.weight"
                            and "lm_head.weight" in params
                            and "lm_head.weight" not in loaded
                        ):
                            weight_loader = getattr(
                                params["lm_head.weight"],
                                "weight_loader",
                                default_weight_loader,
                            )
                            weight_loader(params["lm_head.weight"], loaded_weight)
                            loaded.add("lm_head.weight")
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded.add(weight_name)
                    if (
                        weight_name == "model.embed_tokens.weight"
                        and "lm_head.weight" in params
                        and "lm_head.weight" not in loaded
                    ):
                        weight_loader = getattr(
                            params["lm_head.weight"],
                            "weight_loader",
                            default_weight_loader,
                        )
                        weight_loader(params["lm_head.weight"], loaded_weight)
                        loaded.add("lm_head.weight")
