import os
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import shapely
from shapely import STRtree
from shapely.geometry import box
from pathlib import Path
import sys
sys.path.insert(0, "/content/pipeline")
from config import SAVE_SHP, CHUNK_SIZE
from utils import Timer, log

def _load_boundary(boundary_path):
    # load boundary from folder containing shp or directly from shp/gpkg file
    p = Path(boundary_path)
    if p.is_dir():
        shps = list(p.glob("*.shp"))
        if not shps:
            raise FileNotFoundError(f"No .shp in: {p}")
        p = shps[0]
        log(f"Boundary folder detected - using: {p.name}")
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    if p.suffix.lower() not in (".gpkg", ".shp"):
        raise ValueError(f"Unsupported format: {p.suffix}")
    return gpd.read_file(str(p))

def _raster_bbox(raster_path, target_crs):
    # return raster bounding box as shapely polygon reprojected to target_crs
    import pyproj
    from shapely.ops import transform as shp_transform

    with rasterio.open(str(raster_path)) as src:
        bounds     = src.bounds
        raster_crs = src.crs

    bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    if raster_crs.to_epsg() != target_crs.to_epsg():
        project = pyproj.Transformer.from_crs(
            raster_crs, target_crs, always_xy=True
        ).transform
        bbox = shp_transform(project, bbox)
    return bbox

def _save(gdf, path, save_shp):
    # save geodataframe to gpkg and optionally to shapefile
    gdf.reset_index(drop=True).to_file(str(path), driver="GPKG")
    if save_shp:
        gdf.to_file(str(path.with_suffix(".shp")))

def _map_one(vector_file, cadastral, output_dir, raster_path=None):
    # assign dominant land cover cluster to each cadastral parcel
    stem       = vector_file.stem.replace("_vector", "")
    inter_path = output_dir / f"{stem}_intersections.gpkg"
    final_path = output_dir / f"{stem}_final.gpkg"

    if inter_path.exists() and final_path.exists():
        return None

    lc  = gpd.read_file(str(vector_file))
    utm = cadastral.estimate_utm_crs()
    ca  = cadastral.to_crs(utm).reset_index(drop=True)
    lc  = lc.to_crs(utm).reset_index(drop=True)

    ca["parcel_fid"] = range(1, len(ca) + 1)
    ca["parcel_m2"]  = ca.geometry.area.astype(np.float64)
    ca = ca[ca["parcel_m2"] >= 1.0].reset_index(drop=True)

    if raster_path is not None:
        bbox         = _raster_bbox(raster_path, utm)
        inside_mask  = ca.geometry.intersects(bbox)
        n_outside    = int((~inside_mask).sum())
        ca_inside    = ca[inside_mask].reset_index(drop=True)
        ca_outside   = ca[~inside_mask].reset_index(drop=True)
        print(f"  Parcels in boundary      : {len(ca):,}")
        print(f"  Outside raster extent    : {n_outside:,}  (labelled DN=-1)")
        print(f"  Parcels inside raster    : {len(ca_inside):,}")
    else:
        ca_inside  = ca.copy()
        ca_outside = gpd.GeoDataFrame(columns=ca.columns, crs=utm)
        print(f"  Total parcels            : {len(ca_inside):,}")

    if ca_inside.empty:
        log("WARNING: no parcels inside raster extent")
        return {"parcels": 0, "features": 0}

    candidates = list(ca_inside.sindex.intersection(lc.total_bounds))
    ca_clipped = ca_inside.iloc[candidates].copy().reset_index(drop=True)

    if ca_clipped.empty:
        log("WARNING: no overlap between image and boundary")
        return {"parcels": 0, "features": 0}

    lc_geoms = lc.geometry.values
    ca_geoms = ca_clipped.geometry.values
    fid_arr  = ca_clipped["parcel_fid"].values

    print("  Running spatial join...")
    tree = STRtree(lc_geoms)
    ca_idxs, lc_idxs = tree.query(ca_geoms, predicate="intersects")
    print(f"  Candidate pairs          : {len(ca_idxs):,}")

    if len(ca_idxs) == 0:
        log("WARNING: no intersecting pairs found")
        return {"parcels": len(ca_clipped), "features": 0}

    lc_matched  = lc_geoms[lc_idxs]
    ca_matched  = ca_geoms[ca_idxs]
    dn_matched  = lc["DN"].values[lc_idxs]
    fid_matched = fid_arr[ca_idxs]

    n_chunks = max(1, len(ca_idxs) // CHUNK_SIZE + 1)
    print(f"  Chunks                   : {n_chunks} x {CHUNK_SIZE:,}")

    all_geom, all_area, all_dn, all_fid = [], [], [], []

    for i in range(0, len(ca_idxs), CHUNK_SIZE):
        sl    = slice(i, i + CHUNK_SIZE)
        inter = shapely.intersection(ca_matched[sl], lc_matched[sl])
        area  = shapely.area(inter)
        mask  = area > 0.5
        if not mask.any():
            continue
        all_geom.extend(inter[mask])
        all_area.extend(area[mask])
        all_dn.extend(dn_matched[sl][mask])
        all_fid.extend(fid_matched[sl][mask])
    gc.collect()

    if not all_fid:
        log("No valid intersections found")
        return {"parcels": len(ca_clipped), "features": 0}

    fid_np  = np.array(all_fid,  dtype=np.int64)
    dn_np   = np.array(all_dn,   dtype=np.int64)
    area_np = np.round(np.array(all_area, dtype=np.float64), 2)

    inter_df     = pd.DataFrame({"parcel_fid": fid_np, "DN": dn_np, "area_m2": area_np})
    parcel_total = inter_df.groupby("parcel_fid")["area_m2"].transform("sum")
    pct_np       = np.round((area_np / parcel_total.values) * 100, 2)

    inter_gdf = gpd.GeoDataFrame(
        {"parcel_fid": fid_np, "DN": dn_np,
         "area_m2": area_np, "percentage": pct_np},
        geometry=all_geom, crs=utm,
    )[["parcel_fid", "DN", "area_m2", "percentage", "geometry"]]
    _save(inter_gdf, inter_path, SAVE_SHP)

    grp = inter_df.groupby(["parcel_fid", "DN"],
                            sort=False)["area_m2"].sum().reset_index()
    dominant = (
        grp.loc[grp.groupby("parcel_fid")["area_m2"].idxmax()]
        .reset_index(drop=True)
        .rename(columns={"area_m2": "dominant_area_m2"})
    )

    all_inside = pd.DataFrame({
        "parcel_fid": ca_inside["parcel_fid"].values,
        "parcel_m2":  ca_inside["parcel_m2"].values,
        "geometry":   ca_inside.geometry.values,
    })
    final = all_inside.merge(dominant, on="parcel_fid", how="left")
    final["percentage"] = (
        (final["dominant_area_m2"] / final["parcel_m2"]) * 100
    ).round(2)

    inside_gdf = gpd.GeoDataFrame(
        {
            "parcel_fid": final["parcel_fid"].astype(int),
            "DN":         final["DN"].fillna(-1).astype(int),
            "area_m2":    final["dominant_area_m2"].fillna(0).round(2),
            "percentage": final["percentage"].fillna(0),
        },
        geometry=final["geometry"].values, crs=utm,
    )[["parcel_fid", "DN", "area_m2", "percentage", "geometry"]]

    if not ca_outside.empty:
        outside_gdf = gpd.GeoDataFrame(
            {
                "parcel_fid": ca_outside["parcel_fid"].astype(int),
                "DN":         -1,
                "area_m2":    0.0,
                "percentage": 0.0,
            },
            geometry=ca_outside.geometry.values, crs=utm,
        )[["parcel_fid", "DN", "area_m2", "percentage", "geometry"]]
        final_gdf = gpd.GeoDataFrame(
            pd.concat([inside_gdf, outside_gdf], ignore_index=True),
            crs=utm,
        )
    else:
        final_gdf = inside_gdf

    _save(final_gdf, final_path, SAVE_SHP)

    gc.collect()
    return {"parcels": len(ca), "features": len(inter_gdf)}

def run_dominant_mapping(results_dir, cadastral_path, raster_path=None):
    # entry point for stage 3 - processes all vector files in results_dir
    cadastral = _load_boundary(cadastral_path)
    vectors   = sorted(f for f in os.listdir(results_dir) if f.endswith("_vector.gpkg"))
    if not vectors:
        log("No vector files found")
        return

    log("Stage 3: Class Mapping")
    print()

    for v in vectors:
        log(f"Processing: {v}")
        with Timer() as t:
            result = _map_one(
                results_dir / v, cadastral, results_dir, raster_path=raster_path
            )

        if result:
            print(f"  Total parcels            : {result['parcels']:,}")
            print(f"  Intersections            : {result['features']:,}")
            print(f"Stage 3 completed in {t.pretty()}")
        else:
            print("  Skipped - already processed")
        print()
        gc.collect()
