"""
Entry point to the application. Initializes folder watcher.
"""
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from managers.folder_watcher import start_folder_watcher
import json
def main():
    # Load configuration
    with open("config/connector_mapping.json", "r") as config_file:
        config = json.load(config_file)

    # Extract monitor path from configuration
    monitor_path = config["connectors"]["local_filesystem"]["config"]["monitor_path"]
    print(f"Monitoring path: {monitor_path}")
    # Start folder watcher
    start_folder_watcher(monitor_path)

if __name__ == "__main__":
    main()