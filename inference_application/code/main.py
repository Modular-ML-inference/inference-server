# import required libraries & proto defn.
import grpc
from concurrent import futures
import inference_application.code.protocompiled.basic_inference_pb2_grpc as basic_inf

# import servicer
from inference_application.code.services.basic_inference_svc import BasicInferenceService

def serve():
    # initialize server with 4 workers
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    # attach servicer method to the server
    basic_inf.add_BasicInferenceServiceServicer_to_server(BasicInferenceService(), server)

    # start the server on the port 50051
    server.add_insecure_port("0.0.0.0:50051")
    server.start()
    print("Started gRPC server: 0.0.0.0:50051")

    # server loop to keep the process running
    server.wait_for_termination()


# invoke the server method
if __name__ == "__main__":
    serve()