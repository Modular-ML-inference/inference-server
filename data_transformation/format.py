from abc import ABC, abstractmethod
from data_transformation.exceptions import BadDataFormatException


class DataFormat(ABC):
    schema = {}
    schema_id = ""

    def __init__(self, format, format_file="format.json"):
        self.format_file = format_file
        if self.validate_format(instance=format, schema=self.schema):
            self.format = format
        else:
            raise BadDataFormatException(format=format, schema=self.schema, schema_id=self.schema_id)

    @abstractmethod
    def get_value(self, value_name):
        """A method which returns the value of a given field"""

    @abstractmethod
    def set_value(self, value_name, value):
        """A method which sets the value of a given format field"""

    @abstractmethod
    def validate_format(self, instance, schema):
        """A method which checks whether the format complies with the predefined schema"""
