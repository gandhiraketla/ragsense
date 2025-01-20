import json
import sys
import os
from datetime import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from connectors.data_connector_base import DataSourceConnector
from kafkamanager.kafka_producer import KafkaProducerManager

class LocalFileSystemConnector(DataSourceConnector):
    """
    Connector for monitoring local file system folders.
    """
    def identify_new_data(self, file_path):
                timestamp = datetime.now()
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                message = {
                    "source": "local_filesystem",
                    "data_id": file_path,
                    "metadata": {
                        "timestamp": timestamp_str
                    }
                }
               # self.producer.send(self.kafka_topic, json.dumps(message).encode("utf-8"))
                print(f"Pushed to Kafka: {message}")
                KafkaProducerManager().push_to_topic(message)

    def read(self, data_id):
        """
        Fetch the file content using the file path (data_id).
        """
        with open(data_id, 'r') as file:
            return file.read()
