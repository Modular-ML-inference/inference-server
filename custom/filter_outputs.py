import numpy as np

from typing import List, Dict

from data_transformation.transformation import DataTransformation
from datamodels.models import MachineCapabilities


class FilterOutputs(DataTransformation):
    import numpy as np
    import tensorflow as tf
    id = "fall-detection-p2-filter-outputs"
    description = """A postprocessing step that takes the outputs from the dataset """
    parameter_types = {"conf_thresh": float}
    default_values = {"conf_thresh": 0.5}
    outputs = [Dict[str, np.ndarray]]
    needs = MachineCapabilities(preinstalled_libraries={"numpy": "1.23.5"})

    def set_parameters(self, parameters):
        self.params = parameters

    def get_parameters(self):
        return self.params

    def transform_data(self, data):
        for i in range(len(data)):
            if len(data[i]["scores"]) > 0:
                scores = data[i]["scores"].detach().cpu().numpy()
                srt_det = np.flip(np.argsort(scores))
                srt_det = srt_det[np.where(scores > self.params["conf_thresh"])[0]]

                data[i]['scores'] = data[i]["scores"][srt_det]
                data[i]["masks"] = data[i]["masks"][srt_det]
                data[i]["labels"] = data[i]["labels"][srt_det]
                data[i]["boxes"] = data[i]["boxes"][srt_det]
        return data

    def transform_format(self, format):
        return format