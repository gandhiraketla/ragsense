import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from connectors.local_file_system_connector import LocalFileSystemConnector
from connectors.data_loader import DataLoader
import time

class FolderWatchHandler(FileSystemEventHandler):
    def __init__(self):
        pass

    def on_created(self, event):
        if not event.is_directory:
            print(f"New file detected: {event.src_path}")
            file_path = event.src_path
            test_data = {
                "source": "local_filesystem",
                "data_id": file_path,
                "metadata": {
                    "timestamp": "2024-01-28T12:00:00",
                    "space": ""
                }
            }
            print(test_data)
            loader = DataLoader()
            result = loader.load_data(test_data)


def start_folder_watcher(monitor_path):
    print(f"Starting folder watcher for path: {monitor_path}")
    event_handler = FolderWatchHandler()
    observer = Observer()
    observer.schedule(event_handler, path=monitor_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()