from typing import Dict, Union

from flwr.client import NumPyClient, start_numpy_client, start_client
from flwr.client.dpfedavg_numpy_client import DPFedAvgNumPyClient

from datamodels.models import DPConfiguration, HMConfiguration
from application.src.custom_clients.hm_encryption_client import HMEncryptionClient


class ClientPrivacyManager:

    run_method = staticmethod(start_numpy_client)

    def wrap(self, client: NumPyClient, priv_configuration: Dict[str, Union[HMConfiguration, DPConfiguration]]):
        if "dp-adaptive" in priv_configuration:
            client = self.dp_wrap(client)
        if "homomorphic" in priv_configuration:
            client = self.homomorphic_wrap(client)
            self.run_method = staticmethod(start_client)
        return client

    @staticmethod
    def dp_wrap(client: NumPyClient):
        return DPFedAvgNumPyClient(client)

    @staticmethod
    def homomorphic_wrap(client: NumPyClient):
        return HMEncryptionClient(client)
