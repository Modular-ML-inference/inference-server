from typing import List
from data_transformation.transformation import DataTransformation
from datamodels.models import MachineCapabilities
import numpy as np
import torch

class ImgToTensorTransformation(DataTransformation):
    import torch
    import numpy as np
    """
    Converts a list of images to torch.Tensor previously also scaling it from [0, 255] to [0, 1]
    """
    id = "car-damage-img-to-tensor"
    description = "Converts image to torch.Tensor previously also scaling it from [0, 255] to [0, 1]"
    parameter_types = {"rescale": bool, "use_cuda": bool}
    default_values = {"rescale": True, "use_cuda": True}
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
        if self.params["rescale"]:
            data = [d/255 for d in data] 
        data = [torch.Tensor(d) for d in data]
        is_cuda = self.params["use_cuda"] and torch.cuda.is_available()
        device = "cuda" if is_cuda else "cpu"
        data = [d.to(device) for d in data]
        return data

    def transform_format(self, format):
        if "numerical" in format["data_types"]:
            format["data_types"]["torch-tensor"] = format["data_types"]["numerical"]
        format["data_types"].pop("numerical")
        return format
