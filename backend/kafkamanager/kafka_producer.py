import json
import sys
import os
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from util.envutils import EnvUtils

class KafkaProducerManager:
    """
    A class to manage Kafka producer operations for publishing messages to topics.
    """
    
    def __init__(self):
        """
        Initialize KafkaProducerManager with configuration from environment variables.
        """
        self.env_utils = EnvUtils()
        self.kafka_topic = self.env_utils.get_env('Kafka_topic')
        self.kafka_url = self.env_utils.get_env('Kafka_url')
        self.producer = None
        self.logger = logging.getLogger(__name__)
        self._initialize_producer()

    def _initialize_producer(self):
        """
        Initialize the Kafka producer with the configured settings.
        """
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=[self.kafka_url],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka producer: {str(e)}")
            raise

    def push_message_kafka_topic(self, message):
        """
        Publish a JSON message to the configured Kafka topic.
        
        Args:
            message (dict): The message to be published as JSON
            
        Returns:
            bool: True if message was published successfully, False otherwise
            
        Raises:
            ValueError: If message is not a valid dictionary
            KafkaError: If there's an error publishing to Kafka
        """
        if not isinstance(message, dict):
            raise ValueError("Message must be a dictionary that can be converted to JSON")

        try:
            future = self.producer.send(self.kafka_topic, message)
            # Wait for the message to be delivered
            record_metadata = future.get(timeout=10)
            
            self.logger.info(
                f"Successfully published message to topic {self.kafka_topic} "
                f"partition {record_metadata.partition} "
                f"offset {record_metadata.offset}"
            )
            return True

        except KafkaError as e:
            self.logger.error(f"Failed to publish message to Kafka: {str(e)}")
            return False
        
        except Exception as e:
            self.logger.error(f"Unexpected error while publishing message: {str(e)}")
            return False

    def close(self):
        """
        Close the Kafka producer connection.
        """
        if self.producer:
            self.producer.close()
            self.logger.info("Kafka producer connection closed")