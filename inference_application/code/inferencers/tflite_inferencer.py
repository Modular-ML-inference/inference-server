from inference_application.code.inferencers.base_inferencer import BaseInferencer
import os
import numpy as np
import tensorflow as tf


class TFLiteInferencer(BaseInferencer):

    def load_model(self, path):
        for p in os.listdir(path):
            # check if current path is a file
            if os.path.isfile(os.path.join(path, p)):
                model = tf.lite.Interpreter(model_path=os.path.join(path, p))
                break
        return model

    def prepare(self, inferencer, format):
        self._format = format
        self.inferencer = inferencer
        self.inferencer.allocate_tensors()
        self.input_details = self.inferencer.get_input_details()
        self.output_details = self.inferencer.get_output_details()

    def format(self):
        return self._format

    def predict(self, data):
        data = np.array(data, dtype=np.float32)
        self.inferencer.set_tensor(self.input_details[0]['index'], data)
        self.inferencer.invoke()
        return self.inferencer.get_tensor(self.output_details[0]['index'])
