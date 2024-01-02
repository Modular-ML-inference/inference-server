import numpy as np
from data_transformation.transformation import DataTransformation
from datamodels.models import MachineCapabilities


class ReadImageBatch(DataTransformation):
    from typing import Tuple
    import numpy as np

    id = "read-image-batch"
    description = """A transformation that reads a whole batch of images as ndarray and returns a list"""
    parameter_types = {"axis":int}
    default_values = {"axis": 0}
    outputs = [np.ndarray]
    needs = MachineCapabilities(preinstalled_libraries={"numpy": "1.23.5"})

    def set_parameters(self, parameters):
        self.params = parameters

    def get_parameters(self):
        return self.params

    def transform_data(self, data):
        data = np.array(data)
        batch_size = data.shape[self.params["axis"]]
        data = np.split(data, batch_size, axis=self.params["axis"])
        data = [np.squeeze(d) for d in data]
        return data

    def transform_format(self, format):
        if "numerical" in format["data_types"]:
            format["data_types"]["list"] = {}
            format["data_types"]["list"]["image"] = format["data_types"]["numerical"]
        format["data_types"].pop("numerical")
        return format