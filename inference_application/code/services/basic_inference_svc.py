from inference_application.code.protocompiled import basic_inference_pb2, basic_inference_pb2_grpc
import uuid
from datetime import datetime
from inference_application.code.inference_manager import InferenceManager
from inference_application.code.utils import deconstruct_shape, reconstruct_shape

class BasicInferenceService(basic_inference_pb2_grpc.BasicInferenceServiceServicer):
    ''' 
    This servicer will handle the reading of incoming, basic inference data
    '''

    def __init__(self):
        # initialize inference manager, load model and transformations
        self.inference_manager = InferenceManager()

    def predict(self, request_iterator, context):
        entry_info = dict()
        for request in request_iterator:
            data, shape = request.tensor.array, request.tensor.shape
            inference_data = reconstruct_shape(data, shape)
            print(inference_data)
            input_data = self.inference_manager.transformation_pipeline.transform_data(inference_data)
            # Here, we will call the inference manager and get the global inferencer for prediction
            prediction = self.inference_manager.inferencer.predict(input_data)
            # Return the prediction in a proper format

            print(prediction)
            prediction_data, shape = deconstruct_shape(prediction)
            tensor = basic_inference_pb2.Tensor32(array = prediction_data, shape = shape)
            response = basic_inference_pb2.BasicInferenceResponse(id=int(request.id), 
                                                                  tensor = tensor)
            # stream the response back
            yield response

