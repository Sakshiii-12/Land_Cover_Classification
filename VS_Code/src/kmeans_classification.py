import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import sys
sys.path.insert(0, "/content/pipeline")
from config import (
    NODATA_LABEL, K_MIN, K_MAX, SAMPLE_SIZE,
    SIL_SAMPLE, BATCH_SIZE, BAND_ORDER,
)
from utils import Timer, save_elbow_plot, save_k_scores_csv, is_band_valid, log

def _detect_bands(image, n_bands, desc, ci, band_order):
    def _find(keys, target):
        for i, d in enumerate(desc):
            if d and any(k in d.lower() for k in keys):
                return i
        for i, c in enumerate(ci):
            if c == target:
                return i
        return None

    r   = _find(["red"],             ColorInterp.red)
    g   = _find(["green"],           ColorInterp.green)
    b   = _find(["blue"],            ColorInterp.blue)
    nir = _find(["nir", "infrared"], ColorInterp.undefined)

    if None in (r, g, b):
        order   = band_order.upper().replace(" ", "")
        mapping = {}
        for idx, ch in enumerate(order):
            if ch not in mapping:
                mapping[ch] = idx
        r   = mapping.get("R")
        g   = mapping.get("G")
        b   = mapping.get("B")
        nir = mapping.get("N")

    if nir is not None and nir < n_bands:
        nir = nir if is_band_valid(image[nir]) else None
    else:
        nir = None
    return r, g, b, nir

def _build_features(image, r, g, b, nir):
    from scipy.ndimage import uniform_filter
    eps = 1e-6
    features, names = [], []

    for i in range(image.shape[0]):
        features.append(image[i].ravel())
        names.append(f"Band_{i + 1}")

    features.append(image.mean(axis=0).ravel())
    names.append("Mean_Brightness")

    R   = image[r].ravel()   if r   is not None else None
    G   = image[g].ravel()   if g   is not None else None
    B   = image[b].ravel()   if b   is not None else None
    NIR = image[nir].ravel() if nir is not None else None
    R2D = image[r]           if r   is not None else None
    G2D = image[g]           if g   is not None else None
    B2D = image[b]           if b   is not None else None

    if all(v is not None for v in (R, G, B)):
        features.append(2 * G - R - B);                       names.append("ExG")
        features.append((G - R) / (G + R - B + eps));         names.append("VARI")
        features.append((2*G - R - B) / (2*G + R + B + eps)); names.append("GLI")
        features.append(G / (R + G + B + eps));                names.append("GreenRatio")

    if all(v is not None for v in (R, B)):
        features.append(B / (R + eps)); names.append("BR")

    if all(v is not None for v in (R, G)):
        features.append((G - R) / (G + R + eps)); names.append("NGRDI")
        features.append(1.4 * R - G);             names.append("ExR")

    if all(v is not None for v in (R2D, G2D, B2D)):
        maxc = np.maximum(np.maximum(R2D, G2D), B2D)
        diff = maxc - np.minimum(np.minimum(R2D, G2D), B2D) + eps
        hue  = np.zeros_like(maxc)
        mr, mg, mb = maxc == R2D, maxc == G2D, maxc == B2D
        hue[mr] = ((G2D[mr] - B2D[mr]) / diff[mr]) % 6
        hue[mg] = (B2D[mg] - R2D[mg]) / diff[mg] + 2
        hue[mb] = (R2D[mb] - G2D[mb]) / diff[mb] + 4
        features.append((hue / 6.0).ravel()); names.append("Hue")

    if G2D is not None:
        gn = G2D / (G2D.max() + eps)
        features.append(np.sqrt(np.maximum(
            uniform_filter(gn**2, 5) - uniform_filter(gn, 5)**2, 0
        )).ravel()); names.append("Texture_G")

    if R2D is not None:
        rn = R2D / (R2D.max() + eps)
        features.append(np.sqrt(np.maximum(
            uniform_filter(rn**2, 5) - uniform_filter(rn, 5)**2, 0
        )).ravel()); names.append("Texture_R")

    if all(v is not None for v in (NIR, R)):
        features.append((NIR - R) / (NIR + R + eps));               names.append("NDVI")
        features.append(((NIR - R) / (NIR + R + 0.5 + eps)) * 1.5); names.append("SAVI")
        if G is not None:
            features.append((G - NIR) / (G + NIR + eps));           names.append("MNDWI")

    X = np.column_stack(features)
    col_means = np.nanmean(X, axis=0)
    bad = ~np.isfinite(X)
    X[bad] = np.take(col_means, np.where(bad)[1])
    return X, names

def _classify_one(img_path, output_dir, plot_dir, band_order):
    # run the full classification pipeline for a single image
    out_path   = output_dir / (img_path.stem + "_classified.tif")
    plot_path  = plot_dir   / f"{img_path.stem}_k_selection.png"
    csv_path   = plot_dir   / f"{img_path.stem}_k_scores.csv"

    if out_path.exists() and plot_path.exists() and csv_path.exists():
        import csv as _csv
        with open(str(csv_path)) as f:
            for row in _csv.DictReader(f):
                if row["selected"] == "YES":
                    return int(row["k"]), None, None, None, None
        return None, None, None, None, None

    with rasterio.open(str(img_path)) as src:
        image   = src.read().astype(np.float32)
        profile = src.profile
        desc    = src.descriptions or [None] * src.count
        ci      = src.colorinterp
        nodata  = src.nodata

    bands, height, width = image.shape

    valid_mask = (
        ~np.any(image == nodata, axis=0).ravel()
        if nodata is not None
        else np.ones(height * width, dtype=bool)
    )

    r, g, b, nir = _detect_bands(image, bands, desc, ci, band_order)

    band_labels = []
    for label, idx in [("B", b), ("G", g), ("R", r)]:
        if idx is not None:
            band_labels.append(f"{label}={idx + 1}")
    band_labels.append(f"NIR={nir + 1}" if nir is not None else "NIR=auto-dead")

    X_all, feature_names = _build_features(image, r, g, b, nir)
    X_valid = StandardScaler().fit_transform(X_all[valid_mask])
    rng     = np.random.default_rng(42)
    fit_idx = rng.choice(len(X_valid), min(SAMPLE_SIZE, len(X_valid)), replace=False)
    pca     = PCA(n_components=0.99, random_state=42).fit(X_valid[fit_idx])
    X_pca   = pca.transform(X_valid)

    search_idx = rng.choice(len(X_pca), min(SAMPLE_SIZE, len(X_pca)), replace=False)
    search     = X_pca[search_idx]
    sil_sub    = search[rng.choice(len(search), min(SIL_SAMPLE, len(search)), replace=False)]

    ks, wcss, sil = list(range(K_MIN, K_MAX + 1)), [], []
    for k in ks:
        km = MiniBatchKMeans(n_clusters=k, batch_size=BATCH_SIZE, random_state=42)
        km.fit(search)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(sil_sub, km.predict(sil_sub)))

    kneedle = KneeLocator(ks, wcss, curve="convex", direction="decreasing", S=0.5)
    best_k  = kneedle.knee or ks[0]

    save_elbow_plot(ks, wcss, sil, best_k, img_path.stem, plot_dir)
    save_k_scores_csv(ks, wcss, sil, best_k, img_path.stem, plot_dir)

    km_final = MiniBatchKMeans(
        n_clusters=best_k, batch_size=BATCH_SIZE, random_state=42
    ).fit(search)
    labels = km_final.predict(X_pca)

    full = np.full(height * width, NODATA_LABEL, dtype=np.uint8)
    full[valid_mask] = labels.astype(np.uint8)
    profile.update(count=1, dtype=rasterio.uint8, nodata=NODATA_LABEL)
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(full.reshape(height, width), 1)

    band_str = f"{bands} bands [{', '.join(band_labels)}]"
    return best_k, wcss[ks.index(best_k)], sil[ks.index(best_k)], band_str, feature_names


def run_classification(raw_dir, results_dir, plot_dir, image_configs):
    # entry point for stage 1 - processes all tif/tiff files in raw_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(raw_dir.glob("*.tif")) + sorted(raw_dir.glob("*.tiff"))
    if not images:
        log(f"No .tif images found in {raw_dir}")
        return None

    log("Stage 1: KMeans Classification")
    print()

    best_k_out = None
    for img_path in images:
        cfg        = image_configs.get(img_path.name, {})
        band_order = cfg.get("band_order", BAND_ORDER)

        log(f"Processing: {img_path.name}")
        with Timer() as t:
            best_k, wcss_val, sil_val, band_info, feats = _classify_one(
                img_path, results_dir, plot_dir, band_order
            )

        if best_k is not None:
            best_k_out = best_k

        if wcss_val is not None:
            print(f"  Bands      : {band_info}")
            print(f"  Features   : {len(feats)}")
            print(f"  best_k = {best_k} | WCSS = {wcss_val:,.2f} | Sil = {sil_val:.4f}")
            print(f"  Output     : {img_path.stem}_classified.tif")
            print(f"Stage 1 completed in {t.pretty()}")
        else:
            print(f"  Skipped - already classified  (k = {best_k})")
        print()

    return best_k_out
