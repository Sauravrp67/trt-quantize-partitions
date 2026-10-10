"""Prepare v1 backbones for loading a complete, local detector checkpoint."""
from __future__ import annotations

import torch.nn as nn


class RegNet(nn.Module):
    """Upstream RegNet-Y-040 layout without its unconditional hub download.

    Preserve the upstream ``backbone.model.*`` state-dict keys and hidden-state
    selection. The architecture matches upstream's fixed RegNet-Y-040 model.
    """

    def __init__(self, configuration=None, return_idx=(0, 1, 2, 3)):
        super().__init__()
        from transformers import RegNetConfig, RegNetModel

        self.model = RegNetModel(RegNetConfig(
            num_channels=3, embedding_size=32,
            hidden_sizes=[128, 192, 512, 1088], depths=[2, 6, 12, 2],
            groups_width=64, layer_type="y", hidden_act="relu",
            downsample_in_first_stage=True,
        ))
        self.return_idx = return_idx

    def forward(self, images):
        return self.model(images, output_hidden_states=True).hidden_states[2:5]


def prepare_backbones(config: dict) -> None:
    for name in ("PResNet", "DLANet"):
        if name in config:
            config[name]["pretrained"] = False
    if "RegNet" in config:
        from src.core.yaml_utils import GLOBAL_CONFIG, extract_schema

        # Replace only the constructor registration; upstream source stays intact.
        GLOBAL_CONFIG["RegNet"] = extract_schema(RegNet)
        # Upstream stores the literal string "RegNetConfig()" here, which its
        # constructor ignores. The local constructor has the same fixed layout.
        config["RegNet"]["configuration"] = None
