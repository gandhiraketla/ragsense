import json
import importlib
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from connectors.local_file_system_connector import LocalFileSystemConnector
from connectors.data_connector_base import DataSourceConnector
class DataLoader:
    def __init__(self):
        """Initialize DataLoader with connector configuration"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'connector_mapping.json')
        with open(config_path, 'r') as f:
            self.connector_config = json.load(f)

    def load_data(self, json_data: dict):
        """
        Load data using the appropriate connector based on source
        """
        source = json_data['source']
        connector_info = self.connector_config['connectors'][source]
        
        # Import module and get class directly using module and class_name
        module = importlib.import_module(connector_info['module'])
        connector_class = getattr(module, connector_info['class_name'])
        
        # Create connector instance and load data
        connector = connector_class()
        return connector.load_data(json_data)


# Example usage
if __name__ == "__main__":
    # Test data
    test_data = {
        "source": "confluence",
        "data_id": "1867795",
        "metadata": {
            "timestamp": "2024-01-28T12:00:00",
            "space": "~71202039d865b29006441a9b87f545b2b19e93"
        }
    }
    loader = DataLoader()
    result = loader.load_data(test_data)