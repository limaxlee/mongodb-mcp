import os
import yaml
import argparse
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings

from common.constants import ROOT_DIR

_ENV_MAP = {
    "server_port": ("SERVER_PORT", int),
    "mongodb_host": ("MONGODB_HOST", str),
    "mongodb_port": ("MONGODB_PORT", int),
    "mongodb_db_name": ("MONGODB_DB_NAME", str)
}

_REQUIRED_KEYS = ("server_port", "mongodb_host", "mongodb_port", "mongodb_db_name")


def load_config() -> dict:
    default_config = os.path.join(ROOT_DIR, "config.yaml")
    parser = argparse.ArgumentParser(description="MongoDB MCP Server Configurations")
    parser.add_argument("--config", "-c", type=str, help="Path to the config.yaml", default=default_config)
    args, _ = parser.parse_known_args()

    config = {}
    if os.path.isfile(args.config):
        with open(args.config, "r") as f:
            config.update(yaml.safe_load(f))

    for key, (env_name, caster) in _ENV_MAP.items():
        env_value = os.getenv(env_name)
        if env_value is not None:
            config[key] = caster(env_value)

    missing_config = [key for key in _REQUIRED_KEYS if key not in config or config[key] in (None, "")]
    if missing_config:
        raise ValueError(f"Missing required configuration: {missing_config}")

    return config


class DataDriftConfig(BaseModel):
    """Every tunable of the data drift analysis, loaded from the data_drift section of config.yaml

    The values are echoed back in every analysis result for reproducibility.
    """
    model_config = ConfigDict(frozen=True)

    # Range limits
    max_total_days: int = 30
    max_buckets: int = 60

    # Minimum records per bucket, used by the automatic bucket choice and for merging small buckets
    min_bucket_records_cls: int = 200
    min_bucket_records_det: int = 100
    small_bucket_share: float = 0.3

    # Minimum records on each side of a split before divergence measures are trusted
    min_side_records_cls: int = 500
    min_side_records_det: int = 300
    min_side_boxes: int = 300

    # Minimum number of buckets for each kind of analysis; the change point needs a full segment on each side
    min_buckets_changepoint: int = 4
    min_buckets_trend: int = 6
    min_segment_buckets: int = 2

    # Fixed histogram edges, mandatory so that histograms are comparable across buckets
    confidence_bin_edges: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    area_log_bin_edges: tuple[float, ...] = (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0)
    boxes_per_image_max_bin: int = 5

    # Divergence thresholds
    eps: float = 1e-4
    psi_moderate: float = 0.10
    psi_significant: float = 0.25
    ks_d_geometry: float = 0.15
    ks_p_geometry: float = 0.01
    chi2_p: float = 0.001
    class_prop_change: float = 0.05
    threshold_pressure_ratio: float = 2.0
    threshold_pressure_abs: float = 0.02

    # Trend thresholds
    trend_p: float = 0.05
    trend_tau: float = 0.5
    trend_value_change: float = 0.03
    trend_rate_abs: float = 0.02
    trend_rate_rel: float = 0.5
    trend_count_rel: float = 0.2

    # Outlier threshold
    robust_z: float = 3.5

    # Misc
    near_threshold_margin: float = 0.05
    elapsed_ratio_high: float = 1.5
    elapsed_ratio_low: float = 0.67
    decimals: int = 4
    max_parse_error_examples: int = 10


class Settings(BaseSettings, extra="allow"):
    server_port: int
    mongodb_host: str = "localhost"
    mongodb_port: int = 27017
    mongodb_db_name: str = "default"

    data_drift: DataDriftConfig = DataDriftConfig()


SETTINGS = Settings(**load_config())
