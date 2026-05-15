# Land Cover Classification Using Unsupervised Learning


## Overview

This project presents an end-to-end geospatial image processing and land cover classification pipeline developed for large-scale cadastral parcel analysis using multispectral satellite imagery. The pipeline combines machine learning, remote sensing, and geospatial analysis techniques to automate parcel-level land cover identification over large geographic regions.

The workflow includes:

- Unsupervised land cover classification using MiniBatch K-Means clustering
- Spectral and texture-based feature extraction
- Raster-to-vector polygon conversion
- Parcel-level dominant land cover assignment
- Spatial intersection analysis

The system was developed and tested on approximately **250 km² of Gandhinagar, Gujarat, India** using multispectral LISS-4 imagery and cadastral parcel datasets. The pipeline was also tested on additional multispectral satellite datasets, demonstrating its adaptability to different imagery inputs and study areas.

The repository includes both Google Colab and VS Code implementations of the pipeline. Both implementations contain the same processing workflow and functionality.


## Study Area

| Parameter | Details |
|---|---|
| Location | Gandhinagar, Gujarat, India |
| Approximate Area Covered | 250 km² |
| Input Data | Multispectral Satellite Imagery (LISS-4) |
| Processing Type | Unsupervised Land Cover Classification |
| Boundary Dataset | Cadastral Parcel Boundaries |


The study area consists of urban, vegetation, barren land, and mixed land cover regions across Gandhinagar. Multispectral LISS-4 imagery was used to capture spectral variations between different surface types and support parcel-level land cover analysis.



## Key Features


- MiniBatch K-Means based unsupervised land cover classification
- PCA-based dimensionality reduction for efficient clustering
- Spectral, vegetation, and texture feature extraction
- Raster-to-vector polygon conversion
- Parcel-level dominant class mapping
- Chunk-based spatial intersection processing
- Automatic CRS reprojection for accurate area calculations
- Support for RGB and RGB + NIR imagery
- GeoPackage and optional shapefile export
- Compatible with both Google Colab and VS Code environments



## Repository Structure

```text
Land-Cover-Classification-Pipeline/
│
├── Colab/                                  # Google Colab implementation
│
├── VS_Code/                                # Local system / VS Code implementation
│   ├── config.py                           # Configuration parameters and file paths
│   ├── main.py                             # Main execution script
│   ├── utils.py                            # Utility and helper functions
│   │
│   ├── src/                                # Core processing modules
│   │   ├── __init__.py
│   │   ├── kmeans_classification.py        # Land cover classification
│   │   ├── vectorize_raster.py             # Raster-to-vector conversion
│   │   └── class_mapping.py                # Parcel-level class mapping
│   │
│   └── data/
│       ├── raw/                            # Input imagery
│       ├── plots/                          # K-selection plots
│       └── results/                        # Generated outputs
│
├── cadastral_boundary/                     # Parcel boundary datasets
├── requirements.txt
└── README.md
```



## Pipeline Stages

- Stage 1: Land Cover Classification
- Stage 2: Raster Vectorization
- Stage 3: Parcel Class Mapping

## Classification Features

The feature extraction process adapts dynamically based on the available image bands. To improve land cover separability, the pipeline combines spectral information, vegetation indices, color descriptors, and texture-based features extracted from the input imagery.

### Features for RGB + NIR Imagery

The pipeline generates approximately **20+ features** for 4-band imagery.

| Category | Features |
|---|---|
| Spectral Bands | Blue, Green, Red, NIR |
| Vegetation Indices | NDVI, SAVI, NGRDI, ExG, VARI |
| Water Detection | MNDWI |
| Color Features | Hue, Green Ratio |
| Texture Features | Local Mean, Variance, Entropy, Edge Information |



### Features for RGB Imagery

For RGB imagery, NIR-dependent indices are skipped automatically.

| Category | Features |
|---|---|
| Spectral Bands | Blue, Green, Red |
| Vegetation Features | ExG, VARI, NGRDI |
| Color Features | Hue, Green Ratio |
| Texture Features | Local Mean, Variance, Entropy, Edge Information |

Skipped NIR-dependent indices:

- NDVI
- SAVI
- MNDWI


## Classification Configuration

The pipeline applies PCA before clustering to reduce feature redundancy and computational complexity while preserving approximately 99% variance. Multiple cluster values are automatically evaluated using the Elbow Method and Silhouette Score validation.

For the current study area, the optimal cluster value selected was K =5


### Default Parameters

| Parameter | Default Value | Description |
|---|---|---|
| `K_MIN` | 2 | Minimum number of clusters |
| `K_MAX` | 13 | Maximum number of clusters |
| `SAMPLE_SIZE` | 100000 | Pixels sampled for clustering |
| `SIL_SAMPLE` | 5000 | Pixels sampled for silhouette evaluation |
| `BATCH_SIZE` | 20000 | MiniBatch processing size |
| `NODATA_LABEL` | 255 | NoData pixel label |
| `CONNECTIVITY` | 4 | Raster connectivity |
| `CHUNK_SIZE` | 500000 | Spatial intersection chunk size |
| `SAVE_SHP` | False | Optional shapefile export |

Outputs are primarily stored in the GeoPackage (`.gpkg`) format.

Optional shapefile export can be enabled using:

```python
SAVE_SHP = True
```

## Input and Output Data

### Input Data

The pipeline requires the following input datasets:

- Multispectral satellite imagery (`.tif`)
- Cadastral parcel boundaries (`.shp` or `.gpkg`)
- Band configuration information (RGB or RGB + NIR)

The primary study area was processed using multispectral LISS-4 imagery and cadastral parcel datasets from the Gandhinagar region.


### Generated Outputs

The pipeline generates the following outputs during different processing stages:

### Classification Outputs
- `classified.tif`    Pixel-level classified raster output
- `k_selection.png`   Elbow curve and silhouette score visualization
- `k_scores.csv`      Numerical clustering evaluation metrics

### Vectorization Outputs
- `vector.gpkg`       Vectorized land cover polygons

### Parcel Mapping Outputs
- `intersections.gpkg`   Parcel intersection results
- `final.gpkg`           Final parcel-level land cover mapping

### Output Visualizations
<img width="1006" height="332" alt="Image" src="https://github.com/user-attachments/assets/a5c544be-825c-4782-b0cf-aabe62783248" />



## How to Run the Project

### Running on Google Colab

1. Upload `Pipeline.ipynb`, input imagery, and boundary datasets to Google Drive.

2. Open the notebook in Google Colab.

3. Run the dependency installation cells.

4. Mount Google Drive when prompted.

5. Configure the image and boundary file paths.

Example:

```python
SOURCE_IMAGE = "/content/drive/MyDrive/Gandhinagar.tiff"
BOUNDARY_FILE = "/content/drive/MyDrive/Boundary"
```

6. Run all notebook cells sequentially to execute the complete pipeline.


### Running on VS Code / Local System

1. Clone the repository and navigate to the project directory.

```bash
git clone <repository-link>
cd Land-Cover-Classification-Pipeline
```

2. Create and activate a virtual environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies.

```bash
pip install -r requirements.txt
```

4. Configure the input image and boundary file paths inside `config.py`.

```python
SOURCE_IMAGE = Path("data/raw/image.tif")
BOUNDARY_FILE = Path("data/raw/boundary")
```

5. Execute the pipeline.

```bash
python main.py
```

## Libraries, Frameworks, and Technologies

| Category | Libraries / Technologies | Version |
|---|---|---|
| Programming Language | Python | 3.9+ |
| Geospatial Processing | rasterio | 1.3+ |
| Geospatial Processing | geopandas | 0.14+ |
| Geometry Operations | shapely | 2.0+ |
| Numerical Computing | numpy | 1.24+ |
| Data Analysis | pandas | 2.0+ |
| Machine Learning | scikit-learn | 1.3+ |
| Visualization | matplotlib | 3.7+ |
| Scientific Computing | scipy | 1.11+ |
| Geospatial I/O | fiona | 1.9+ |
| Coordinate Systems | pyproj | 3.6+ |
| Spatial Indexing | rtree | 1.1+ |
| Clustering Analysis | kneed | 0.8+ |
| Development Environment | Google Colab | Cloud Environment |
| Development Environment | VS Code | Local Environment |
