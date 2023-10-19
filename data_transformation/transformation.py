from abc import ABC, abstractmethod
from typing import Dict, Any, List

from datamodels.models import MachineCapabilities


class DataTransformation(ABC):
    """
    A class to represent a singular data transformation.
    ...
    Attributes
    ----------
    id : str
        The unique id of a given data transformation, by way of which it can be obtained from the database
    description : str
        A written, detailed description of what is the purpose of a given data transformation
    parameter_types : Dict[str, type]
        The dictionary defining what kind of parameters (aside from data) can be accepted by the transformation
    default_values : Optional[str, Any]
        An optional dictionary defining the default values of the (selected) parameters
    outputs : List[type]
        A list detailing the types of outputs of transformation
    needs : MachineCapabilities
        A set of needed capabilities compliant with MachineCapabilities base model
    """
    id: str = None
    description: str = "A basic data transformation template"
    parameter_types: Dict[str, type] = {}
    default_values: Dict[str, Any] = None
    outputs: List[type] = None
    needs: MachineCapabilities = MachineCapabilities()

    def __init__(self):
        self.params = self.default_values

    @abstractmethod
    def set_parameters(self, parameters):
        """Set the data transformation to use specific parameter values"""

    @abstractmethod
    def get_parameters(self):
        """Get the parameter values defined for the transformation"""

    @abstractmethod
    def transform_data(self, data):
        """Transform the data according to the description"""

    @abstractmethod
    def transform_format(self, format):
        """Transform the data format according to the set description"""
