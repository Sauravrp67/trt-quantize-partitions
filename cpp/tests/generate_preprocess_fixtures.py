"""Regenerate small Pillow reference tensors used by the native parity test."""
from pathlib import Path

import numpy as np
from PIL import Image

out = Path(__file__).parent / 'data'
out.mkdir(exist_ok=True)
for width, height in [(23, 17), (11, 23), (9, 11)]:
    out_width = 23 if width == 9 else 9
    out_height = 17 if height == 11 else 11
    y, x = np.mgrid[:height, :width]
    bgr = np.stack([(x * 17 + y * 13) % 256,
                    (x * 7 + y * 31) % 256,
                    (x * 43 + y * 3) % 256], axis=-1).astype(np.uint8)
    rgb = Image.fromarray(bgr[..., ::-1], 'RGB')
    resized = np.asarray(rgb.resize((out_width, out_height), Image.Resampling.BILINEAR))
    (out / f'pillow_{width}x{height}_to_{out_width}x{out_height}.bin').write_bytes(
        resized[..., ::-1].tobytes())
