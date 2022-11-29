from typing import Dict, Union

from flwr.client import NumPyClient
from flwr.client.dpfedavg_numpy_client import DPFedAvgNumPyClient

from application.datamodels.models import DPConfiguration, HMConfiguration


class ClientPrivacyManager:

    def wrap(self, client: NumPyClient, priv_configuration: Dict[str, Union[HMConfiguration, DPConfiguration]]):
        if "dp-adaptive" in priv_configuration:
            strategy = self.dp_wrap(client)
        elif "homomorphic" in priv_configuration:
            strategy = self.homomorphic_wrap(client)
        return strategy

    @staticmethod
    def dp_wrap(client: NumPyClient):
        return DPFedAvgNumPyClient(client)

    @staticmethod
    def homomorphic_wrap(client: NumPyClient):
        return client