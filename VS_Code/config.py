from pathlib import Path

# Root directory of the pipeline
BASE_DIR = Path("/content/pipeline")

# Input file paths on Google Drive
SOURCE_IMAGE  = Path("ADD_PATH")
BOUNDARY_FILE = Path("ADD_PATH")

# Output directories
RAW_DIR     = BASE_DIR / "data/raw"
PLOT_DIR    = BASE_DIR / "data/plots"
RESULTS_DIR = BASE_DIR / "data/results"

# Band order string
BAND_ORDER = "BGRNIR"

# Kmeans search range and pixel sampling limits
K_MIN       = 2
K_MAX       = 13
SAMPLE_SIZE = 100_000
SIL_SAMPLE  = 5_000
BATCH_SIZE  = 20_000

# Raster and vector processing settings
NODATA_LABEL = 255
CONNECTIVITY = 4
CHUNK_SIZE   = 500_000
SAVE_SHP     = False
