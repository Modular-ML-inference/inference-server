import numpy as np
from data_transformation.transformation import DataTransformation
from datamodels.models import MachineCapabilities


class ResizeNumpyArray(DataTransformation):
    from typing import Tuple
    import numpy as np

    id = "car-damage-resize-transformation"
    description = """A transformation that rescale all of the input values according to the multiplier. Here, the multiplier is used to """
    parameter_types = {"target_size":Tuple[int]}
    # here the values are in order of x, y, z, svm
    default_values = {"target_size":[1200, 900]}
    outputs = [np.ndarray]
    needs = MachineCapabilities(preinstalled_libraries={"numpy": "1.23.5"})

    def set_parameters(self, parameters):
        self.params = parameters

    def get_parameters(self):
        return self.params

    def transform_data(self, data):
        data = [data] if not isinstance(data, list) else data
        data = [np.array(d) for d in data]
        return [np.resize(d, self.params["target_size"]) for d in data]

    def transform_format(self, format):
        if "image" in format["data_types"]:
            format["data_types"]["image"]["size"] = self.params["target_size"]
        return format
