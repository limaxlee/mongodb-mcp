import os
import yaml
import argparse
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


class Settings(BaseSettings, extra="allow"):
    server_port: int
    mongodb_host: str = "localhost"
    mongodb_port: int = 27017
    mongodb_db_name: str = "default"


SETTINGS = Settings(**load_config())
