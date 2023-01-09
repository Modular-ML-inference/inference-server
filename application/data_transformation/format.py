from abc import ABC, abstractmethod
from jsonschema import validate
from application.data_transformation.exceptions import BadDataFormatException


class DataFormat(ABC):
    schema = {}
    schema_id = ""

    def __init__(self, format):
        if self.validate(instance=format, schema=self.schema):
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
    def validate_format(self, format):
        """A method which checks whether the format complies with the predefined schema"""


class CarScannerDataFormat(DataFormat):
    schema_id = "default_twotronics"
    schema = {
        "type": "object",
        "properties": {
            "price": {"type": "number"},
            "name": {"type": "string"},
        },
    }

    def get_value(self, value_name):
        return self.format[value_name]

    def set_value(self, value_name, value):
        self.format[value_name] = value

    def validate_format(self, format):
        validate(instance=format)
