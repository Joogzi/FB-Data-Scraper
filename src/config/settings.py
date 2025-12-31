"""Configuration management for the application."""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class MetricConfig:
    """Configuration for a single metric extractor."""
    enabled: bool = True
    roi: Optional[Dict[str, int]] = None  # {x1, y1, x2, y2}
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    """Application configuration."""
    # Video settings
    video_path: str = ""
    output_dir: str = "output"
    
    # Metric configurations
    speed: MetricConfig = field(default_factory=MetricConfig)
    gforce: MetricConfig = field(default_factory=MetricConfig)
    torque_fl: MetricConfig = field(default_factory=MetricConfig)
    torque_fr: MetricConfig = field(default_factory=MetricConfig)
    torque_rl: MetricConfig = field(default_factory=MetricConfig)
    torque_rr: MetricConfig = field(default_factory=MetricConfig)
    
    # OCR settings
    use_gpu: bool = True
    
    # Export settings
    export_format: str = "csv"  # csv, json
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Create from dictionary."""
        # Convert nested dicts to MetricConfig
        for key in ["speed", "gforce", "torque_fl", "torque_fr", "torque_rl", "torque_rr"]:
            if key in data and isinstance(data[key], dict):
                data[key] = MetricConfig(**data[key])
        return cls(**data)


class ConfigManager:
    """Manages application configuration persistence."""
    
    DEFAULT_CONFIG_NAME = "config.json"
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to config file in current directory
            config_path = self.DEFAULT_CONFIG_NAME
        self.config_path = Path(config_path)
        self._config: Optional[AppConfig] = None
    
    @property
    def config(self) -> AppConfig:
        """Get current configuration, loading from file if needed."""
        if self._config is None:
            self._config = self.load()
        return self._config
    
    def load(self) -> AppConfig:
        """Load configuration from file."""
        if not self.config_path.exists():
            return AppConfig()
        
        try:
            with open(self.config_path, "r") as f:
                if self.config_path.suffix == ".yaml":
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            return AppConfig.from_dict(data)
        except Exception as e:
            print(f"Error loading config: {e}")
            return AppConfig()
    
    def save(self, config: Optional[AppConfig] = None) -> bool:
        """Save configuration to file."""
        config = config or self._config or AppConfig()
        self._config = config
        
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                if self.config_path.suffix == ".yaml":
                    yaml.safe_dump(config.to_dict(), f, default_flow_style=False)
                else:
                    json.dump(config.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def update(self, **kwargs) -> None:
        """Update configuration values."""
        config = self.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        self._config = config
    
    def get_metric_config(self, metric_name: str) -> Optional[MetricConfig]:
        """Get configuration for a specific metric."""
        config = self.config
        return getattr(config, metric_name, None)
    
    def set_metric_config(self, metric_name: str, metric_config: MetricConfig) -> None:
        """Set configuration for a specific metric."""
        if hasattr(self.config, metric_name):
            setattr(self._config, metric_name, metric_config)
