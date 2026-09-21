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

    # Range limits: total span of both windows and the number of periods the automatic granularity aims for
    max_total_days: int = 400
    max_periods: int = 60

    # Minimum predictions (boxes for detection) on each side of a split before divergence measures are trusted
    min_side_records_cls: int = 500
    min_side_records_det: int = 300
    min_side_boxes: int = 300

    # Minimum number of periods for the change point search and on each side of a split
    min_periods_changepoint: int = 4
    min_segment_periods: int = 2

    # Divergence thresholds
    eps: float = 1e-4
    psi_moderate: float = 0.10
    psi_significant: float = 0.25
    chi2_p: float = 0.001
    class_prop_change: float = 0.05
    threshold_pressure_ratio: float = 2.0
    threshold_pressure_abs: float = 0.02

    # Misc
    decimals: int = 4


class Settings(BaseSettings, extra="allow"):
    server_port: int
    mongodb_host: str = "localhost"
    mongodb_port: int = 27017
    mongodb_db_name: str = "default"

    data_drift: DataDriftConfig = DataDriftConfig()


SETTINGS = Settings(**load_config())
