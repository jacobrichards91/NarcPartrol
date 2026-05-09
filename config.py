# All tuneable knobs in one place — no magic numbers in stage files.

# ---------------------------------------------------------------------------
# GPS Segmentation (S2)
# ---------------------------------------------------------------------------
MIN_LOT_FRONTAGE_M: float = 15.0   # meters of travel before starting a new house
                                    # Tighten for dense urban rows; loosen for rural

# ---------------------------------------------------------------------------
# Frame Sampling (S3)
# ---------------------------------------------------------------------------
SAMPLE_FPS: int = 4                 # frames per second extracted from the video

# ---------------------------------------------------------------------------
# Building Detection (S4)  —  YOLOv8x trained on Open Images V7
# ---------------------------------------------------------------------------
YOLO_MODEL: str = "yolov8x-oiv7.pt"
BUILDING_CONF_MIN: float = 0.30     # minimum detection confidence to accept a building
BUILDING_COVERAGE_MIN: float = 0.15 # building bbox must cover ≥ this fraction of frame area

# Open Images V7 label names we treat as "the building we want"
BUILDING_CLASS_NAMES: frozenset = frozenset({
    "Building", "House", "Tower", "Skyscraper", "Shed",
})

# OIV7 label names whose bounding boxes over the building incur an occlusion penalty
OCCLUDER_CLASS_NAMES: frozenset = frozenset({
    "Car", "Truck", "Bus", "Van", "Person", "Tree", "Motorcycle",
})

# ---------------------------------------------------------------------------
# Quality Scoring (S5)
# ---------------------------------------------------------------------------
TOP_N_CANDIDATES: int = 3          # frames per house segment sent to OCR and possibly cloud

# Score = w_sharp*sharpness + w_cov*coverage + w_front*frontality + w_exp*exposure
#       - OCCLUSION_PENALTY_WEIGHT * occlusion_fraction
W_SHARPNESS:  float = 0.30
W_COVERAGE:   float = 0.25
W_FRONTALITY: float = 0.25
W_EXPOSURE:   float = 0.20
OCCLUSION_PENALTY_WEIGHT: float = 0.50

# ---------------------------------------------------------------------------
# Address / OCR (S6)
# ---------------------------------------------------------------------------
OSM_SEARCH_RADIUS_M: int = 75      # Overpass API radius around segment GPS centroid
OSM_TIMEOUT_S: int = 10            # HTTP timeout for Overpass queries
# Minimum OCR confidence to accept a detected number string
OCR_CONF_MIN: float = 0.40

# ---------------------------------------------------------------------------
# Cloud Quality Gate (S7)
# ---------------------------------------------------------------------------
QUALITY_THRESHOLD: float = 0.45    # invoke Claude if top frame score < this
CLOUD_MODEL: str = "claude-haiku-4-5-20251001"
CLOUD_MAX_IMAGE_BYTES: int = 5 * 1024 * 1024   # 5 MB cap before resizing for API

# ---------------------------------------------------------------------------
# Export (S8)
# ---------------------------------------------------------------------------
CROP_PADDING: float = 0.12         # fractional padding added to building bbox on each side
JPEG_QUALITY: int = 95
OUTPUT_SUBDIR: str = "snapshots"
LOG_FILENAME: str = "log.csv"
