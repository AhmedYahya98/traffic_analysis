import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and manage project configuration from YAML files."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration parameters
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['model', 'input', 'tracking', 'analysis', 'output']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'model.confidence')
            default: Default value if key not found
        
        Returns:
            Configuration value or default        
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self.config['model']
    
    def get_input_config(self) -> Dict[str, Any]:
        """Get input configuration section."""
        return self.config['input']
    
    def get_tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration section."""
        return self.config['tracking']
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration section."""
        return self.config['analysis']
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration section."""
        return self.config.get('visualization', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration section."""
        return self.config['output']