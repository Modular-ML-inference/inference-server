import flwr as fl
from starlette.concurrency import run_in_threadpool
from application.config import ORCHESTRATOR_ADDRESS, default_twotronics_config
from application.additional.utils import ModelLoader
from application.config import FEDERATED_PORT
from application.src.builders.keras_builder import KerasBuilder
from application.src.builders.pytorch_builder import PytorchBuilder
from application.src.privacy_manager import ClientPrivacyManager

current_jobs = {}


async def start_client(training_id, config):
    loader = ModelLoader()
    library = loader.check_library(config.model_name, config.model_version)
    if library == "pytorch":
        builder = PytorchBuilder(training_id, default_twotronics_config)
    else:
        builder = KerasBuilder(training_id, config)
    unsafe_client = builder.product()
    client = ClientPrivacyManager().wrap(unsafe_client, config.privacy_mechanisms)
    await run_in_threadpool(
        lambda: fl.client.start_numpy_client(server_address=f"{config.server_address}:{FEDERATED_PORT}", client=client))
    if training_id in current_jobs and current_jobs[training_id] > 1:
        current_jobs[training_id] -= 1
    else:
        current_jobs.pop(training_id)
