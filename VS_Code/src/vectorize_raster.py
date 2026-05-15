import os
import gc
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import sys
sys.path.insert(0, "/content/pipeline")
from config import CONNECTIVITY, NODATA_LABEL
from utils import Timer, log

def _vectorize_one(raster_path, output_dir):
    # convert a single classified raster to polygon gpkg
    out_path = output_dir / (raster_path.stem.replace("_classified", "_vector") + ".gpkg")
    if out_path.exists():
        return out_path, None

    with rasterio.open(str(raster_path)) as src:
        raster    = src.read(1)
        transform = src.transform
        crs       = src.crs

    records = [
        {"geometry": shape(geom), "DN": int(val)}
        for geom, val in shapes(raster, transform=transform, connectivity=CONNECTIVITY)
    ]
    gdf = gpd.GeoDataFrame(records, crs=crs)

    # remove nodata polygons before saving
    gdf = gdf[gdf["DN"] != NODATA_LABEL].reset_index(drop=True)
    gdf.insert(0, "FID", range(1, len(gdf) + 1))
    gdf = gdf[["FID", "DN", "geometry"]]
    gdf.to_file(str(out_path), driver="GPKG")

    n = len(gdf)
    del records, gdf
    gc.collect()
    return out_path, n

def run_vectorization(results_dir):
    # entry point for stage 2 - vectorises all classified rasters in results_dir
    rasters = sorted(f for f in os.listdir(results_dir) if f.endswith("_classified.tif"))
    if not rasters:
        log("No classified rasters found")
        return

    log("Stage 2: Raster to Vector Conversion")
    print()

    for r in rasters:
        log(f"Processing: {r}")
        with Timer() as t:
            out_path, n_polys = _vectorize_one(results_dir / r, results_dir)

        if n_polys is not None:
            print(f"  Polygons : {n_polys:,}")
            print(f"  Output   : {out_path.name}")
            print(f"Stage 2 completed in {t.pretty()}")
        else:
            print(f"  Skipped - already vectorised")
        print()
        gc.collect()
