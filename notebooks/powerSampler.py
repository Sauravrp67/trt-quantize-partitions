import sys; from pathlib import Path
sys.path.insert(0, str(Path.cwd().resolve().parents[0]))   # if the notebook is in notebooks/
from harness.power import PowerSampler
import pynvml, torch

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
print("right now: ", pynvml.nvmlDeviceGetPowerUsage(h) / 1000)

a = torch.randn(4096, 4096, device="cuda"); b = torch.randn(4096, 4096, device="cuda")
N = 200

with PowerSampler(interval_s = 0.01) as ps:
    for _ in range(N):
        c = a @ b
    
    torch.cuda.synchronize()

s = ps.stats
print(f"mean {s.mean_w:.1f} W | peak {s.peak_w:.1f} W | "
      f"energy {s.energy_j:.1f} J | {s.energy_j/N*1000:.2f} mJ per call")