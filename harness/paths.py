from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = REPO_ROOT / "data" / "coco"
COCO_VAL_IMAGES = COCO_DIR / "val2017"
COCO_VAL_ANN = COCO_DIR / "annotations" / "instances_val2017.json"
ENGINES_DIR = REPO_ROOT / "results" / "engines"
TABLES_DIR = REPO_ROOT / "results" / "tables"