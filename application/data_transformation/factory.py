from abc import ABC, abstractmethod
from typing import Dict, Any, List


class DataTransformationPipeline(ABC):
    """
    A class used to perform, in order, a set of data transformations
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
