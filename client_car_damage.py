# import required libraries & proto defn.
import json
import grpc
import grpc.experimental
import numpy as np
import tensorflow as tf
from PIL import Image
from inference_application.code.protocompiled import extended_inference_pb2, extended_inference_pb2_grpc

# initialize channel to gRPC server
#channel = grpc.insecure_channel(target="localhost:50052")
options = [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
with grpc.insecure_channel('localhost:50052', options=options) as channel:
    greeter_stub = extended_inference_pb2_grpc.ExtendedInferenceServiceStub(channel)
    path = './damage_detection/car damage segmentation.v3i.coco-segmentation/'
    with open(path+'_more_images_for_inference.json', 'r') as file:
        final_images_to_take = json.load(file)
    batch = []
    size = 4
    np.random.seed(42)
    sample_img = np.random.choice(np.array(final_images_to_take), size = size, replace = False)
    for i in range(size):
        img = Image.open(path+sample_img[i]['file_name']).convert("RGB").resize((1200, 900))
        batch.append(img)
    values = np.array([np.array(b) for b in batch])
    tensor = tf.make_tensor_proto(values = values, shape=[4,900,1200,3])
    request_iterator = iter(
        [
            extended_inference_pb2.ExtendedInferenceRequest(id=1, input={"array": tensor})
        ]
    )

    response_iterator = greeter_stub.predict(request_iterator, 10)
    for response in response_iterator:
        print(tf.make_ndarray(response.output["results"]))

    # You should get results [2 0 3 0] for the first dataset, I will also prepare version of this for the new dataset ASAP