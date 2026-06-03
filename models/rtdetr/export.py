"""RT-DETR model export entrypoint placeholder.

Planned role:
    Export an RT-DETR PyTorch checkpoint to a static batch=1 FP32 ONNX model
    consumed by the shared pipeline.

TODO:
    - Load the selected RT-DETR implementation and checkpoint.
    - Freeze preprocessing and dynamic-shape assumptions.
    - Export FP32 ONNX with stable input/output names.
    - Emit metadata needed by graph surgery and benchmark stages.
"""
from pathlib import Path
import sys
import types
import torch
import torch.nn as nn
import argparse

REPO_ROOT = Path(__file__).resolve().parents[2]
RTDETR_PYTORCH_ROOT = REPO_ROOT / "RT-DETR" / "rtdetr_pytorch"
RTDETR_SRC_ROOT = RTDETR_PYTORCH_ROOT / "src"
if not RTDETR_SRC_ROOT.is_dir():
    raise FileNotFoundError(f"RT-DETR PyTorch checkout not found: {RTDETR_PYTORCH_ROOT}")

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RTDETR_PYTORCH_ROOT))

# RT-DETR's top-level src/__init__.py eagerly imports data modules that are not
# needed for ONNX export and are tied to older torchvision beta APIs.
src_pkg = types.ModuleType("src")
src_pkg.__file__ = str(RTDETR_SRC_ROOT / "__init__.py")
src_pkg.__path__ = [str(RTDETR_SRC_ROOT)]
sys.modules["src"] = src_pkg

from src.core import YAMLConfig
import src.nn  # noqa: F401 - registers model/backbone classes for YAMLConfig
import src.zoo  # noqa: F401 - registers RT-DETR and postprocessor classes
from harness import verify_parity

class RTDETRExportModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = config.model.deploy()
    
    def forward(self, images):
        outputs = self.model(images)
        return outputs
    
def export(args,):
    config = YAMLConfig(args.config, resume = args.ckpt)

    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location = 'cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    
    config.model.load_state_dict(state)

    data = torch.rand(1,3,640,640)
    model = RTDETRExportModel(config=config)
    _ = model(data)

    dynamic_axes = {
        'images': {0: 'batch_size'}
    }
    
    onnx_program = torch.onnx.export(
        model,
        (data,),
        input_names = ['images'],
        output_names=['pred_logits', 'pred_boxes'],
        dynamic_shapes=dynamic_axes,
        dynamo = True,
        verbose = False,
        verify=True,
        profile=True,
        keep_initializers_as_inputs=False,
        report = True,
        dump_exported_program=True,
        artifacts_dir = "./artifacts"
    )

    onnx_program.initialize_inference_session()    
    print(verify_parity(onnx_program._inference_session, model = model))
    
    
    # onnx_program.save(args.file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--ckpt', '-r', type=str, )
    parser.add_argument('--file-name', '-f', type=str, default='model.onnx')
    parser.add_argument('--check',  action='store_true', default=False,)
    parser.add_argument('--simplify',  action='store_true', default=False,)

    args = parser.parse_args()

    export(args)
