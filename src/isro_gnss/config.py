from __future__ import annotations

import yaml


class Config(dict):
    """Dict that also supports dot access, recursively, for nested YAML config."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        return Config(yaml.safe_load(f))
