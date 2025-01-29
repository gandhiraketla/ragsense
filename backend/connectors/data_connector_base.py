from abc import ABC, abstractmethod

class DataSourceConnector(ABC):
    """
    Abstract base class for all data source connectors.
    """

    @abstractmethod
    
    def load_data(self,json_data: dict):
        """
        Identify new data in the source and push metadata to Kafka.
        """
        pass

   
