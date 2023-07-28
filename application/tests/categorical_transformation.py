
from data_transformation.transformation import DataTransformation
from datamodels.models import MachineCapabilities
import numpy as np
import tensorflow
from tensorflow.keras.utils import to_categorical

class CategoricalTransformation(DataTransformation):
    """
    A class to represent the categorical transformation
    """
    id = "categorical_transformation"
    description = "This class transforms y data into categorical data"
    parameter_types = {"categories":int}
    default_values = {"categories": 10}
    outputs = [np.ndarray, np.ndarray]
    needs = MachineCapabilities()

    def __init__(self):
        self.params = self.default_values

    def set_parameters(self, parameters):
        """Set the data transformation to use specific parameter values"""
        self.params = parameters

    def get_parameters(self):
        """Get the parameter values defined for the transformation"""
        return self.params

    def transform_data(self, data):
        """Transform the data according to the description"""
        (x, y) = data
        y = to_categorical(y, self.params["categories"])
        return (x, y)

    def transform_format(self, format):
        """Transform the data format according to the set description"""
        return format