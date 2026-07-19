from pathlib import Path
import sys
import types
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
RTDETR_PYTORCH_ROOT = REPO_ROOT / "RT-DETR" / "rtdetr_pytorch"
RTDETR_SRC_ROOT = RTDETR_PYTORCH_ROOT / "src"
if not RTDETR_SRC_ROOT.is_dir():
    raise FileNotFoundError(f"RT-DETR PyTorch checkout not found: {RTDETR_PYTORCH_ROOT}")

sys.path.insert(0, str(RTDETR_PYTORCH_ROOT))

# RT-DETR's top-level src/__init__.py eagerly imports data modules that are not
# needed for ONNX export and are tied to older torchvision beta APIs.
src_pkg = types.ModuleType("src")
src_pkg.__file__ = str(RTDETR_SRC_ROOT / "__init__.py")
src_pkg.__path__ = [str(RTDETR_SRC_ROOT)]
sys.modules["src"] = src_pkg

from src.core import YAMLConfig
import src.nn
import src.zoo


class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = config.model.deploy()

    def forward(self, images):
        outputs = self.model(images)
        return outputs


def main():
    config = YAMLConfig(str(RTDETR_PYTORCH_ROOT / "configs/rtdetr/rtdetr_r18vd_6x_coco.yml"), resume = str(REPO_ROOT / "models/rtdetr/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"))

    model = Model(config = config).to("cuda")
    images = torch.randn(3, 3, 640, 640, device = "cuda")

    startEvent = torch.cuda.Event(enable_timing=True)
    endEvent = torch.cuda.Event(enable_timing=True)


    startEvent.record()
    outputs = model(images)
    endEvent.record()
    torch.cuda.synchronize()

    time_elapsed = startEvent.elapsed_time(end_event=endEvent)
    
    print(outputs.keys())
    print(time_elapsed)

if __name__ == "__main__":
    main()


        
