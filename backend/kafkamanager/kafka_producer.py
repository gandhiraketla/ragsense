from kafka import KafkaProducer as BaseKafkaProducer
from json import dumps
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from util.envutils import EnvUtils
import logging

class KafkaProducerManager:
    def __init__(self):
        self.env_utils = EnvUtils()
        self.kafka_topic = self.env_utils.get_env('Kafka_topic')
        self.kafka_url = self.env_utils.get_env('Kafka_url')
        
        try:
            self.producer = BaseKafkaProducer(
                bootstrap_servers=[self.kafka_url],
                value_serializer=lambda x: dumps(x).encode('utf-8')
            )
        except Exception as e:
            logging.error(f"Failed to create Kafka producer: {str(e)}")
            raise

    def push_to_topic(self, json_data):
        try:
            future = self.producer.send(self.kafka_topic, value=json_data)
            # Block until message is sent or timeout occurs
            future.get(timeout=10)
            logging.info(f"Successfully sent message to topic {self.kafka_topic}")
            return True
        except Exception as e:
            logging.error(f"Failed to send message to Kafka: {str(e)}")
            return False
        
    def __del__(self):
        if hasattr(self, 'producer'):
            self.producer.close()