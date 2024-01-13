from inference_application.code.protocompiled import extended_inference_pb2, extended_inference_pb2_grpc
from prometheus_client import Summary, Histogram
from inference_application.code.inference_manager import InferenceManager
from inference_application.code.main import inference_lock

INFERENCE_REQUEST_TIME = Summary(
    'inference_request_processing_seconds', 'Time spent processing request')
h = Histogram('inference_request_latency_seconds',
              'Histogram for request processing of ML inference')


class ExtendedInferenceService(extended_inference_pb2_grpc.ExtendedInferenceServiceServicer):
    ''' 
    This is a new, extended version of the basic servicer 
    that enables the service to consume different kinds of data through the use of 
    tensorflow tensors
    '''

    def __init__(self):
        # initialize inference manager, load model and transformations
        self.inference_manager = InferenceManager()

    @INFERENCE_REQUEST_TIME.time()
    @h.time()
    def predict(self, request_iterator, _):
        for request in request_iterator:
            with inference_lock:
                input_data = self.inference_manager.preprocessing_pipeline.transform_data(
                    request)
                # Here, we will call the inference manager and get the global inferencer for prediction
                prediction = self.inference_manager.inferencer.predict(
                    input_data)
                output = self.inference_manager.postprocessing_pipeline.transform_data(
                    prediction)
                response = extended_inference_pb2.ExtendedInferenceResponse(id=int(request.id),
                                                                            output=output)
                # stream the response back
                yield response
