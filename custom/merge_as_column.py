from data_transformation.transformation import DataTransformation
from datamodels.models import MachineCapabilities


import numpy as np


from typing import List


class MergeAsColumn(DataTransformation):
    import numpy as np

    id = "fall-detection-p2-merge-as-column"
    description = """A transformation that takes a list and merges it as an additional data column"""
    parameter_types = {"merge_axis": int, "list_axis": int}
    default_values = {"merge_axis": 1, "list_axis": 1}
    outputs = [np.ndarray]
    needs = MachineCapabilities(preinstalled_libraries={"numpy": "1.23.5"})

    def set_parameters(self, parameters):
        self.params = parameters

    def get_parameters(self):
        return self.params

    def transform_data(self, data):
        data, lengths = data
        data, lengths = np.array(data), np.array(lengths)
        lengths = np.expand_dims(lengths, axis=self.params["list_axis"])
        data = np.append(data, lengths, axis=self.params["merge_axis"])
        return data

    def transform_format(self, format):
        if "numerical" in format["data_types"]:
            format["data_types"]["numerical"]["size"][self.params["merge_axis"]] += 1
        return format
