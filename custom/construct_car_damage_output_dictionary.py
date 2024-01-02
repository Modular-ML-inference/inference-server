from typing import List

import tensorflow as tf
from data_transformation.transformation import DataTransformation
from datamodels.models import MachineCapabilities
import numpy as np
import torch

class ConstructCarDamageOutputDictionary(DataTransformation):
    import numpy as np
    from typing import Dict
    """
    This is a transformation that takes the output from the car damage detection RCNN model and transforms 
    it appropriately to a dictionary of tensorflow tensors, that may then be analyzed by people.
    """
    id = "car-damage-construct-car-output"
    description = "This is a transformation that takes the output from the car damage detection RCNN model and transforms it to a readable dictionary."
    parameter_types = {}
    default_values = {}
    outputs = [Dict[str,np.ndarray]]
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
        """Transform the output according to the description"""
        boxes = []
        masks = []
        results = np.array([], dtype = np.int32)
        labels = np.array([], dtype = np.int64)
        scores = np.array([], dtype = np.float32)
        for output in data:
            num_detections = len(output['scores'])
            results = np.append(results, num_detections)
            if num_detections > 0:
                [boxes.append(box) for box in output['boxes'].detach().to('cpu').numpy()]
                [masks.append(mask) for mask in output['masks'].detach().to('cpu').numpy()]
                labels = np.append(labels, output['labels'].detach().to('cpu').numpy())
                scores = np.append(scores, output['scores'].detach().to('cpu').numpy())
                
        boxes = np.array(boxes, dtype = np.float32)
        masks = np.array(masks, dtype = np.uint8)
        results = tf.make_tensor_proto(values=results)
        labels = tf.make_tensor_proto(values=labels)
        boxes = tf.make_tensor_proto(values=boxes)
        scores = tf.make_tensor_proto(values=scores)
        masks = tf.make_tensor_proto(values=masks)
        return {'results': results, 'labels': labels, 'boxes': boxes, 'scores': scores, 'masks': masks}

    def transform_format(self, format):
        return format