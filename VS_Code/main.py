import sys
import shutil
import time
from pathlib import Path
sys.path.insert(0, "/content/pipeline")
from config import (
    SOURCE_IMAGE, BOUNDARY_FILE,
    RAW_DIR, RESULTS_DIR, PLOT_DIR, BAND_ORDER,
)
from utils import Timer, fmt_time, log
from src.kmeans_classification import run_classification
from src.vectorize_raster      import run_vectorization
from src.class_mapping         import run_dominant_mapping


def _validate():
    # check all required input files exist before starting
    errors = []

    if not SOURCE_IMAGE.exists():
        errors.append(f"Image not found - {SOURCE_IMAGE}")
    else:
        log(f"Input ready: {SOURCE_IMAGE.name}")

    p = Path(BOUNDARY_FILE)
    if p.is_dir():
        shps = list(p.glob("*.shp"))
        if not shps:
            errors.append("No .shp file found in boundary folder")
        else:
            log(f"Boundary folder detected - using: {shps[0].name}")
    elif p.suffix.lower() in (".gpkg", ".shp"):
        if not p.exists():
            errors.append(f"Boundary file not found - {p.name}")
        else:
            log(f"Boundary file ready: {p.name}")
    else:
        errors.append(f"Unsupported boundary format - {p.suffix}")

    if errors:
        print()
        for e in errors:
            print(f"[ERROR] {e}")
        sys.exit(1)

def _prepare():
    # create output dirs and copy source image to data/raw if needed
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dst = RAW_DIR / SOURCE_IMAGE.name
    if not dst.exists():
        shutil.copy2(SOURCE_IMAGE, dst)
        log(f"Copied {SOURCE_IMAGE.name} to data/raw/")

def main():
    start = time.time()
    print()
    log(f"Pipeline started at {time.strftime('%Y-%m-%d')}")
    print()
    _validate()
    _prepare()
    print()

    image_configs = {SOURCE_IMAGE.name: {"band_order": BAND_ORDER}}

    with Timer() as t1:
        best_k = run_classification(
            RAW_DIR, RESULTS_DIR, PLOT_DIR, image_configs
        )
    print()

    with Timer() as t2:
        run_vectorization(RESULTS_DIR)
    print()

    classified_path = RESULTS_DIR / f"{SOURCE_IMAGE.stem}_classified.tif"
    with Timer() as t3:
        run_dominant_mapping(RESULTS_DIR, BOUNDARY_FILE, raster_path=classified_path)

    log(f"Total runtime: {fmt_time(time.time() - start)}")
    print()

if __name__ == "__main__":
    main()
