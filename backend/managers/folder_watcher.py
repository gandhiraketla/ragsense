import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from connectors.local_file_system_connector import LocalFileSystemConnector
import time

class FolderWatchHandler(FileSystemEventHandler):
    def __init__(self):
        pass

    def on_created(self, event):
        if not event.is_directory:
            print(f"New file detected: {event.src_path}")
            LocalFileSystemConnector().identify_new_data(event.src_path)


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