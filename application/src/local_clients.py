from logging import INFO

from flwr.common.logger import log

from application.additional.utils import BasicModelLoader
from application.config import FEDERATED_PORT
from application.src.builders.keras_builder import KerasBuilder
from application.src.builders.pytorch_builder import PytorchBuilder
from application.src.privacy_manager import ClientPrivacyManager
from starlette.concurrency import run_in_threadpool

current_jobs = {}


async def start_client(training_id, config):
    loader = BasicModelLoader()
    library = loader.check_library(config.model_name, config.model_version)
    if library == "pytorch":
        builder = PytorchBuilder(training_id, config)
    else:
        builder = KerasBuilder(training_id, config)
    unsafe_client = builder.prepare_training()
    privacy_manager = ClientPrivacyManager()
    client = privacy_manager.wrap(unsafe_client, config.privacy_mechanisms)
    log(INFO,
        f'The client tries to access the server on {config.server_address}:{FEDERATED_PORT}')
    await run_in_threadpool(
        lambda: privacy_manager.run_method(server_address=f"{config.server_address}:{FEDERATED_PORT}", client=client))
    if training_id in current_jobs and current_jobs[training_id] > 1:
        current_jobs[training_id] -= 1
    else:
        current_jobs.pop(training_id)
