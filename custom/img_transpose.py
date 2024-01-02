from typing import List, Tuple
from data_transformation.transformation import DataTransformation
from datamodels.models import MachineCapabilities
import numpy as np

class ImgTransposeTransformation(DataTransformation):
    from typing import Tuple
    """
    Transposes the axis of the image from [H, W, C] to match tensor dimensions [C, H, W]
    """
    id = "car-damage-img-transpose"
    description = "Transposes the axis of the image from [H, W, C] to match tensor dimensions [C, H, W]"
    parameter_types = {"axes": Tuple[int]}
    default_values = {"axes": (2,0, 1)}
    outputs = [np.ndarray, np.ndarray]
    needs = MachineCapabilities()

    def __init__(self):
        self.params = self.default_values
        self.transient_values = {}

    def set_parameters(self, parameters):
        """Set the data transformation to use specific parameter values"""
        self.params = parameters

    def get_parameters(self):
        """Get the parameter values defined for the transformation"""
        return self.params

    def transform_data(self, data):
        """Transform the data according to the description"""
        data = [data] if not isinstance(data, list) else data
        data = [d.transpose(self.params["axes"]) for d in data]
        return data

    def transform_format(self, format):
        if "image" in format["data_types"]["list"]:
            array_as = np.array(format["data_types"]["list"]["image"]["size"])
            rearranged = array_as[list(self.params["axes"])]
            format["data_types"]["list"]["image"]["size"] = list(rearranged)
        return format
