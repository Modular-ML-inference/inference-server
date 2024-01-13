# import required libraries & proto defn.
import time
import grpc
import grpc.experimental
import numpy as np
import tensorflow as tf
from inference_application.code.protocompiled import extended_inference_pb2, extended_inference_pb2_grpc

# initialize channel to gRPC server
# channel = grpc.insecure_channel(target="localhost:50052")
while True:
    with grpc.insecure_channel('0.0.0.0:50052') as channel:
        greeter_stub = extended_inference_pb2_grpc.ExtendedInferenceServiceStub(
            channel)
        values = np.array([-972.0, -187.0, -304.0, -972.0, -187.0, -304.0, -972.0, -187.0, -304.0, -972.0, -187.0, -304.0, -972.0, -
                          187.0, -304.0, -968.0, -230.0, -273.0, -968.0, -175.0, -281.0, -964.0, -218.0, -304.0, -964.0, -187.0, -300.0])*2048.0/8000.0
        values2 = np.array([-1226.0, 117.0,  -726.0, -1226.0, 117.0,  -726.0, -1226.0, 117.0,  -726.0, -1226.0, 117.0,  -726.0, -1226.0,
                           117.0,  -726.0, -1742.0,  -332.0,  -171.0, -1160.0,  -285.0,  -437.0, 54.0, -5910.0, -5113.0, 187.0, -160.0, -1675.0])*2048.0/8000.0
        tensor = tf.make_tensor_proto(
            values=values, shape=[9, 3], dtype=np.intc)
        tensor2 = tf.make_tensor_proto(
            values=values2, shape=[9, 3], dtype=np.intc)
        with open('message_not_fall.bin', 'wb') as f:
            f.write(extended_inference_pb2.ExtendedInferenceRequest(
                id=1, input={"array": tensor}).SerializeToString())
        with open('message_fall.bin', 'wb') as f:
            f.write(extended_inference_pb2.ExtendedInferenceRequest(
                id=2, input={"array": tensor2}).SerializeToString())
        request_iterator = iter(
            [
                extended_inference_pb2.ExtendedInferenceRequest(
                    id=1, input={"array": tensor}),
                extended_inference_pb2.ExtendedInferenceRequest(
                    id=2, input={"array": tensor2})
            ]
        )

        response_iterator = greeter_stub.predict(request_iterator, 10)
        for response in response_iterator:
            print(response.output)
    time.sleep(0.5)

    # You should get confidence of 0.258879334 for the first example and 0.991416454 for the second
