from abc import ABC, abstractmethod

class DataSourceConnector(ABC):
    """
    Abstract base class for all data source connectors.
    """

    @abstractmethod
    def identify_new_data(self):
        """
        Identify new data in the source and push metadata to Kafka.
        """
        pass

    @abstractmethod
    def read(self, data_id):
        """
        Fetch the actual data using the given data identifier.
        """
        pass
