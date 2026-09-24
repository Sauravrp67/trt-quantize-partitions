"""CPU tests for the standalone runtime, including a synthetic COCO evaluation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from deploy.runtime import preprocess, postprocess
from deploy.eval_map import detections_to_coco, evaluate_engine

DEPLOY = Path(__file__).resolve().parents[1]


def test_preprocess_rgb_layout_and_normalization():
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    image[:, :, 2] = 128
    actual = preprocess(image)
    assert actual.shape == (1, 3, 640, 640)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    np.testing.assert_allclose(actual[0, 0], 1)
    np.testing.assert_allclose(actual[0, 1], 0)
    np.testing.assert_allclose(actual[0, 2], 128 / 255)


def test_decode_preserves_low_scores_and_maps_boxes_without_overflow():
    logits = np.full((1, 300, 80), -1000, dtype=np.float32)
    logits[0, 7, 2] = 1000
    boxes = np.zeros((1, 300, 4), dtype=np.float32)
    boxes[0, 7] = [.5, .5, .2, .4]
    with np.errstate(over='raise', invalid='raise'):
        labels, xyxy, scores = postprocess(logits, boxes, (100, 50))
    assert len(scores) == 300
    assert labels[0] == 2 and scores[0] == 1
    assert np.count_nonzero(scores) == 1
    np.testing.assert_allclose(xyxy[0], [40, 15, 60, 35], atol=1e-5)


def test_coco_category_mapping_and_box_conversion():
    row = detections_to_coco(12, [11], [[10, 20, 40, 60]], [.75])[0]
    assert row == {'image_id': 12, 'category_id': 13,
                   'bbox': [10., 20., 30., 40.], 'score': .75}


def test_saved_engine_predictions_reach_coco_eval(tmp_path):
    from pycocotools.coco import COCO
    annotation = tmp_path / 'instances.json'
    annotation.write_text(json.dumps({
        'info': {},
        'images': [{'id': 1, 'file_name': 'one.jpg', 'width': 100, 'height': 100}],
        'categories': [{'id': 1, 'name': 'person'}],
        'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                         'bbox': [25, 25, 50, 50], 'area': 2500, 'iscrowd': 0}],
    }))
    Image.new('RGB', (100, 100)).save(tmp_path / 'one.jpg')
    sessions = []

    class FakeSession:
        def __init__(self, path):
            assert path == Path('existing.plan')
            self.closed = False
            sessions.append(self)

        def run(self, x):
            assert x.shape == (1, 3, 640, 640)
            logits = np.full((1, 300, 80), -100, np.float32)
            logits[0, 0, 0] = 10
            boxes = np.zeros((1, 300, 4), np.float32)
            boxes[0, 0] = [.5, .5, .5, .5]
            return {'pred_logits': logits, 'pred_boxes': boxes}

        def close(self):
            self.closed = True

    result = evaluate_engine(Path('existing.plan'), COCO(str(annotation)), tmp_path,
                             [1], session_factory=FakeSession)
    assert result['mAP50_95'] == pytest.approx(1)
    assert result['mAP50'] == pytest.approx(1)
    assert sessions[0].closed


def test_runtime_folder_works_without_host_or_training_imports(tmp_path):
    copied = tmp_path / 'runtime-only'
    shutil.copytree(DEPLOY, copied, ignore=shutil.ignore_patterns('__pycache__', 'tests'))
    # Fail immediately if any CLI imports host/training libraries, even if installed here.
    guard = tmp_path / 'import-guard'
    guard.mkdir()
    (guard / 'sitecustomize.py').write_text('''
import sys
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'onnxruntime', 'modelopt', 'harness', 'src', 'tensorrt', 'polygraphy'}:
            raise RuntimeError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Guard())
''')
    env = dict(os.environ, PYTHONPATH=str(guard))
    for script in ['infer.py', 'eval_map.py', 'import_model.py']:
        result = subprocess.run([sys.executable, str(copied / script), '--help'],
                                cwd=tmp_path, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert 'usage:' in result.stdout


def test_import_checksum_failure_leaves_no_model(tmp_path):
    source = tmp_path / 'source.onnx'
    source.write_bytes(b'model transfer test')
    output = tmp_path / 'out' / 'model.onnx'
    result = subprocess.run([sys.executable, str(DEPLOY / 'import_model.py'),
                             '--source', str(source), '--out', str(output),
                             '--sha256', '0' * 64], capture_output=True, text=True)
    assert result.returncode != 0 and 'mismatch' in result.stderr
    assert not output.exists()
    assert list(output.parent.iterdir()) == []


def test_model_import_preserves_bytes_and_refuses_overwrite(tmp_path):
    import hashlib
    source = tmp_path / 'source.onnx'
    source.write_bytes(b'local model bytes')
    output = tmp_path / 'models' / 'copy.onnx'
    command = [sys.executable, str(DEPLOY / 'import_model.py'), '--source', str(source),
               '--out', str(output), '--sha256', hashlib.sha256(source.read_bytes()).hexdigest()]
    subprocess.run(command, check=True, capture_output=True)
    assert output.read_bytes() == source.read_bytes()
    source.write_bytes(b'changed')
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and 'destination exists' in result.stderr
    assert output.read_bytes() == b'local model bytes'


def test_build_command_works_from_another_directory(tmp_path):
    executable = tmp_path / 'fake-trtexec'
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    model = tmp_path / 'model with spaces.onnx'
    model.write_bytes(b'fake ONNX for command construction only')
    output = tmp_path / 'engines'
    result = subprocess.run(['bash', str(DEPLOY / 'build_engines.sh'), '-o', str(output),
                             '-w', '1024', str(model), '--', '--noTF32'], cwd=tmp_path,
                            env=dict(os.environ, TRTEXEC=str(executable)),
                            text=True, capture_output=True, check=True)
    for argument in ['--stronglyTyped', '--memPoolSize=workspace:1024', '--noTF32',
                     f'--onnx={model}', f'--saveEngine={output / "model with spaces.plan"}']:
        assert argument in result.stdout.splitlines()
    assert (output / 'build_model with spaces.log').exists()
