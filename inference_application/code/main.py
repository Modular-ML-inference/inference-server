import grpc
from concurrent import futures
import prometheus_client
import threading

from inference_application.code.service_manager import ServiceManager

inference_lock = threading.Lock()

def serve():
    # initialize prometheus server
    prometheus_client.start_http_server(9000)
    options = ('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)
    # initialize server with 4 workers
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=200, options=options))

    # The setup of the service is flexible and can be seen more thoroughly in classes ServiceManager and InferenceSetupLoader
    ServiceManager().setup_service(server)

    # start the server on the port 50051
    server.add_insecure_port("0.0.0.0:50051")
    server.start()
    print("Started gRPC server: 0.0.0.0:50051")

    # server loop to keep the process running
    server.wait_for_termination()


# invoke the server method
if __name__ == "__main__":
    serve()
