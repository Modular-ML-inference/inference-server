import logging
import os
from logging import INFO

import numpy as np
import tenseal as ts
from flwr import common
from flwr.client import Client, NumPyClient
from flwr.common import Code
from flwr.common.typing import Status, GetPropertiesIns, GetPropertiesRes

from application.additional.utils import HMSerializer, ts_tensors_to_parameters, parameters_to_ts_tensors
from application.config import HM_SECRET_FILE


class HMEncryptionClient(Client):
    """Wrapper for configuring a Flower client for homomorphic encryption
    It allows for sending parameters containing CKKS tensors."""

    def __init__(self, client: NumPyClient, file_path=os.path.join(os.sep, "code", "application", "src", "custom_clients", "hm_keys", HM_SECRET_FILE)) -> None:
        super().__init__()
        logger = logging.getLogger()
        # log all messages, debug and up
        logger.setLevel(INFO)
        self.context = ts.context_from(HMSerializer.read(file_path))
        self.client = client

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        config = self.client.get_properties(ins.config)
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties=config,
        )

    def get_parameters(self, config) -> common.GetParametersRes:
        weights = self.client.get_parameters(config)
        v_mid = [ts.plain_tensor(v) if len(v.shape) !=
                 0 else ts.plain_tensor([v]) for v in weights]
        weights = [ts.ckks_tensor(self.context, m) if len(
            m.shape) != 0 else np.array(0) for m in v_mid]
        parameters = ts_tensors_to_parameters(weights)
        return common.GetParametersRes(status=Status(code=Code.OK, message="Success"), parameters=parameters)

    def fit(self,  ins: common.FitIns) -> common.FitRes:
        tensors = parameters_to_ts_tensors(ins.parameters)
        weights = [np.array(tensor.decrypt().tolist(), dtype=np.float32)
                   for tensor in tensors]
        updated_params, num_examples, metrics = self.client.fit(
            weights, ins.config)
        v_mid = [ts.plain_tensor(v) if len(
            v.shape) != 0 else ts.plain_tensor([v]) for v in updated_params]
        v_mid1 = [ts.ckks_tensor(self.context, m) if len(
            m.shape) != 0 else np.array(0) for m in v_mid]
        v_mid = ts_tensors_to_parameters(v_mid1)
        return common.FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=v_mid,
            num_examples=num_examples,
            metrics=metrics
        )

    def evaluate(self, ins: common.EvaluateIns) -> common.EvaluateRes:
        tensors = parameters_to_ts_tensors(ins.parameters)
        weights = [np.array(tensor.decrypt().tolist(), dtype=np.float32)
                   for tensor in tensors]
        loss, num_examples, metrics = self.client.evaluate(weights, ins.config)
        return common.EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=loss,
            num_examples=num_examples,
            metrics=metrics
        )
