# FL Local Operations

The FL Local Operations is an enabler developed as a part of the [Assist-IoT project](https://assist-iot.eu/). 
It was developed as a part of an FL system along with the FL Orchestrator, FL Local Operations and FL Repository and should ideally be deployed with those enablers in order to use its full functionality (although it is possible to conduct an FL training process without using the FL Orchestrator which serves as a GUI, configuring the enabler throught the use of its dedicated REST API). It encapsulates the functionalities of a federated learning (FL) client by maintaining a connection with the FL Orchestrator (GUI and monitoring), connecting to the training initiated by the FL server (FL Training Collector), periodically providing it with local weights and obtaining new global weights, as well as downloading any necessary components from the FL Repository (database).

Beyond the classic functionality of an FL client, however, FL Local Operations also enables the local inference deployment of a selected model (that can function as a standalone container), the use of flexible configurations, basic format verification and pluggable components, as well as selected privacy mechanisms. FL Local Operations is compatible with Prometheus metric monitoring. 

## Helm chart

The FL Local Operations enabler has been developed with the assumption that it will be deployed on a Kubernetes cluster with a dedicated Helm chart. To do so, just go to the `helm-chart` directory and run `helm install fllocaloperationslocal fllocaloperations`. If you want to deploy multiple FL Local Operations in one Kubernetes cluster, just choose different names for all of the deployments. If you want to deploy only the inference component, run `helm install fllocaloperationslocal fllocaloperations --set inferenceapp.fullDeployment.enabled=false`.

To make sure that before that the enabler has been configured properly, check the 3 ConfigMaps that are deployed alongside the enabler. Their names change depending on the name od the deployment (to allow for multiple Local Operations instances to coexist in a Kubernetes cluster while having slightly different configurations). 

The first, which name starts with `flinference-config-map`, serves to flexibly set and change the configuration for the inference component, including the data format received by the gRPC service (as `format.json`), the name, version and input format of the model (as `model.json`), the configuration of the data transformation pipeline (as `transformation_pipeline.json`) and the data about both the serialized gRPC service and the specific inferencer to be used (as `setup.json`).

The second config map, which name begins with `fllocalops-config-map` contains the environmental variables necessary to deploy the FL Local Operations instance. Check especially the fields of `REPOSITORY_ADDRESS` (the address of the nearest FL Repository instance), `ORCHESTRATOR_SVR_ADDRESS` (the address of the FL Orchestrator's main service), `ORCHESTRATOR_WS_ADDRESS` (the address that the websocket should use to connect to the FL Orchestrator) and `SERVER_ADDRESS` (the address of the FL Training Collector). If you change something in the ConfigMap when the enabler is already deployed, destroy the inferenceapp and trainingapp pods to let them recreate with the updated configuration.

Finally, the ConfigMap beginning with `fltraining-config-map` describes the configuration necessary to run the trainingapp component with pluggable transformations. This includes the data format that the data loader has access to (as `format.json`), the input format of the model (as `model.json`), the configuration of the train data transformation pipeline (as `transformation_pipeline_train.json`) as well as test data transformation pipleine (`transformation_pipeline_test.json`) and the data about both the specific data loader and training client that will need to be used (as `setup.json`).

If you'd like to see and experiment with the API, the recommended approach is to go to the http://127.0.0.1:XXXXX/docs URL (if the NodePort for the first FL Local Operations endpoint has been changes, it should be also updated in the URL) and use the Swagger docs generated by the FastAPI framework.

## Docker image

You can run `USER_INDEX=1 FL_LOCAL_OP_DATA_FOLDER="./data" docker compose up --force-recreate --build -d` in your terminal to build a new Docker image or use the `start-local.sh` script to do it automatically (for instance, by running the command `./start-local.sh 1`).  

## Local Training Configuration

In order to initiate the training, a JSON encompassing the following configuration should be sent to the endpoint shown below. The most important available keys and their meaning will be explained further down. 

**POST /job/config/{training_id}/**

```json
{
  "client_type_id": "string",
  "server_address": "string",
  "eval_metrics": [
    "string"
  ],
  "eval_func": "string",
  "num_classes": 0,
  "num_rounds": 0,
  "shape": [
    0
  ],
  "training_id": 0,
  "model_name": "string",
  "model_version": "string",
  "config": [
    {
      "config_id": "string",
      "batch_size": 0,
      "steps_per_epoch": 0,
      "epochs": 0,
      "learning_rate": 0
    }
  ],
  "optimizer_config": {
    "optimizer": "string",
    "lr": 0,
    "rho": 0,
    "eps": 0,
    "foreach": true,
    "maximize": true,
    "lr_decay": 0,
    "betas": [
      "string",
      "string"
    ],
    "etas": [
      "string",
      "string"
    ],
    "step_sizes": [
      "string",
      "string"
    ],
    "lambd": 0,
    "alpha": 0,
    "t0": 0,
    "max_iter": 0,
    "max_eval": 0,
    "tolerance_grad": 0,
    "tolerance_change": 0,
    "history_size": 0,
    "line_search_fn": "string",
    "momentum_decay": 0,
    "dampening": 0,
    "centered": true,
    "nesterov": true,
    "momentum": 0,
    "weight_decay": 0,
    "amsgrad": true,
    "learning_rate": 0,
    "name": "string",
    "clipnorm": 0,
    "global_clipnorm": 0,
    "use_ema": true,
    "ema_momentum": 0,
    "ema_overwrite_frequency": 0,
    "jit_compile": true,
    "epsilon": 0,
    "clipvalue": 0,
    "initial_accumulator_value": 0,
    "beta_1": 0,
    "beta_2": 0,
    "beta_2_decay": 0,
    "epsilon_1": 0,
    "epsilon_2": 0,
    "learning_rate_power": 0,
    "l1_regularization_strength": 0,
    "l2_regularization_strength": 0,
    "l2_shrinkage_regularization_strength": 0,
    "beta": 0
  },
  "scheduler_config": {
    "scheduler": "string",
    "step_size": 0,
    "gamma": 0,
    "last_epoch": 0,
    "verbose": true,
    "milestones": [
      0
    ],
    "factor": 0,
    "total_iters": 0,
    "start_factor": 0,
    "end_factor": 0,
    "monitor": "string",
    "min_delta": 0,
    "patience": 0,
    "mode": "string",
    "baseline": 0,
    "restore_best_weights": true,
    "start_from_epoch": 0,
    "cooldown": 0,
    "min_lr": 0
  },
  "warmup_config": {
    "scheduler": "string",
    "warmup_iters": 0,
    "warmup_epochs": 0,
    "warmup_factor": 0,
    "scheduler_conf": {
      "scheduler": "string",
      "step_size": 0,
      "gamma": 0,
      "last_epoch": 0,
      "verbose": true,
      "milestones": [
        0
      ],
      "factor": 0,
      "total_iters": 0,
      "start_factor": 0,
      "end_factor": 0,
      "monitor": "string",
      "min_delta": 0,
      "patience": 0,
      "mode": "string",
      "baseline": 0,
      "restore_best_weights": true,
      "start_from_epoch": 0,
      "cooldown": 0,
      "min_lr": 0
    }
  },
  "privacy-mechanisms": {
    "homomorphic": {
      "poly_modulus_degree": 8192,
      "coeff_mod_bit_sizes": [
        60,
        40,
        40
      ],
      "scale_bits": 40,
      "scheme": "CKKS"
    },
    "dp-adaptive":{
      "num_sampled_clients": 0,
      "init_clip_norm": 0.1,
      "noise_multiplier": 1,
      "server_side_noising": true,
      "clip_count_stddev": null,
      "clip_norm_target_quantile": 0.5,
      "clip_norm_lr": 0.2
    }
  }
}
```
The definitions:
- **client_type_id**
  Specifies the ID of the client. Allows to bypass the plugability modules for the Pytorch builder with the keyword "base" for testing purposes. 
- **server_address**
  The address of the Flower server that the FL client should try to connect to.
- **eval_metrics**
  The evaluation metrics which will be gathered through the evaluation process by the FL client.
- **eval_func**
  The evaluation function that the model will use as the loss throughout the training process.
- **num_classes**
  The number of classes in classification problems.
- **num_rounds**
  The number of rounds that the training should run for.
- **shape**
  The shape of the data. Currently, this parameter is recommended to be changed through the ConfigMaps instead.
- **training_id**
  The id of the training process being conducted.
- **model_name**
  The name of the model that will be used in the training. The name should be the same as the one stored in FL Repository.
- **model_version**
  The version of the model that will be used in the training. The name should be the same as the one stored in the FL Repository.
- **config**
  The configuration specifying how the FL training process will be conducted on the client, containing important terms such as the batch_size or learning rate.
- **optimizer_config**
  The configuration of the optimizer. 
  - **optimizer**
    For the Keras model and client, the optimizer can be one of:
    ```python
    "sgd": tf.keras.optimizers.SGD,
    "rmsprop": tf.keras.optimizers.RMSprop,
    "adam": tf.keras.optimizers.Adam,
    "adadelta": tf.keras.optimizers.Adadelta,
    "adagrad": tf.keras.optimizers.Adagrad,
    "adamax": tf.keras.optimizers.Adamax,
    "nadam": tf.keras.optimizers.Nadam,
    "ftrl": tf.keras.optimizers.Ftrl
    ```
    For the PyTorch model and client, the optimizer can be one of:
    ```python
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sparseadam": torch.optim.SparseAdam,
    "adamax": torch.optim.Adamax,
    "asgd": torch.optim.ASGD,
    "lbfgs": torch.optim.LBFGS,
    "nadam": torch.optim.NAdam,
    "radam": torch.optim.RAdam,
    "rmsprop": torch.optim.RMSprop,
    "rprop": torch.optim.Rprop,
    "sgd": torch.optim.SGD
    ```
  Other fields indicate the arguments that should be passed to the optimizer.
- **scheduler_config**
  The configuration of the scheduler.
  - **scheduler**
    For the Keras model and client, the scheduler  (or here, a more appropriate name would be a Keras callback) can be one of:
    ```python
    "earlystopping": tf.keras.callbacks.EarlyStopping,
    "reducelronplateau": tf.keras.callbacks.ReduceLROnPlateau,
    "terminateonnan": tf.keras.callbacks.TerminateOnNaN
    ```
    For the Pytorch model and client, the scheduler can be one of:
    ```python
    "lambdalr": torch.optim.lr_scheduler.LambdaLR,
    "multiplicativelr": torch.optim.lr_scheduler.MultiplicativeLR,
    "steplr": torch.optim.lr_scheduler.StepLR,
    "multisteplr": torch.optim.lr_scheduler.MultiStepLR,
    "constantlr": torch.optim.lr_scheduler.ConstantLR,
    "linearlr": torch.optim.lr_scheduler.LinearLR,
    "exponentiallr": torch.optim.lr_scheduler.ExponentialLR,
    "cosineannealinglr": torch.optim.lr_scheduler.CosineAnnealingLR,
    "chainedscheduler": torch.optim.lr_scheduler.ChainedScheduler,
    "sequentiallr": torch.optim.lr_scheduler.SequentialLR,
    "reducelronplateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "cycliclr": torch.optim.lr_scheduler.CyclicLR,
    "onecyclelr": torch.optim.lr_scheduler.OneCycleLR,
    "cosineannealingwarmrestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    ```
    Other fields indicate the arguments that should be passed to the scheduler.
- **warmup_config**
  The configuration of an (optional) warmup. This configuration is valid only for the PyTorch builder. It specifies a special scheduler, which can be used only for a selected number of epochs to provide warmup throughout the process.
  - **scheduler**
    The name of the scheduler.
  Other fields indicate the arguments that should be passed to the scheduler.

- **privacy-mechanisms**
  The configuration indicating which privacy mechanisms should the FL Training Collector employ (if any) and what should be their parameters. This dictionary can have no keys (which indicates no privacy mechanisms used), "homomorphic" which indicates the use of HE, "dp-adaptive" which indicates the use of Differential Privacy with Adaptive Clipping or both "homomorphic" and "dp-adaptive", which indicates that both techniques should be used.
  - **homomorphic** 
    The parametres configurable to be used for homomorphically encrypted federated averaging are used to specify the context as described in the [TenSEAL](https://github.com/OpenMined/TenSEAL) documentation.
  - **dp-adaptive**
    The parametres specifying the differentially private Federated Averaging are taken from the Flower library and, by proxy, from the [relevant paper](https://arxiv.org/pdf/1905.03871.pdf).

A sample test configuration can be seen here:

```json
{"client_type_id": "local1",
  "server_address": "trainingcollectorlocal-trainingmain-svc2",
  "eval_metrics": [
    "accuracy"
  ],
  "eval_func": "categorical_crossentropy",
  "num_classes": 10,
  "num_rounds": 15,
  "shape": [
    32, 32, 3
  ],
  "training_id": "10",
  "model_name": "keras_test",
  "model_version": "version_1",
  "config": [
    {"config_id": "min_effort",
   "batch_size": "64",
   "steps_per_epoch": "32",
   "epochs": "1",
   "learning_rate": "0.001"}
  ],
  "optimizer_config": {
    "optimizer": "adam",
    "learning_rate":"0.005",
    "amsgrad":"True"
  },
  "scheduler_config": {
    "scheduler": "reducelronplateau",
    "factor":"0.5",
    "min_delta":"0.0003"
  },
  "privacy-mechanisms":{}}
```

## Other API endpoints

- **POST /model/**
  Receive new training model metadata for local storage
- **PUT /model/{name}/{version}**
  Update the weights and structure of the locally stored training model.
- **GET /job/status**
  Get the statuses of the current jobs.
- **GET /job/total**
  Get the number of currently running jobs.
- **GET /capabilities**
  Get the computational capabilities of the machine that FL Local Operations is running on.
- **GET /format**
  Get the format of the data that a given FL Local Operations instance has currently access to.

## Websockets

A websocket client is running in the background of the trainingapp pod. Its purpose is to provide a continuous means of communication with the FL Orchestrator, so that the Orchestrator knows exactly which FL Local Operations are active and can participate in training. It will try to connect with the FL Orchestrator server via the `ORCHESTRATOR_WS_ADDRESS` address configured in the `fllocalops-config-map` ConfigMap. To appropriately change it is then enough to modify this address with `kubectl edit cm` and recreate the trainingapp pod.

## The inference component

The inference component corresponds to the inferenceapp pod and can function as a standalone. It uses gRPC for lightweight communication. It allows for the configuration setup through the modification of configuration files located in the `configurations` directory (which can also be modified on the fly by changing the values in the `flinference-config-map` and restarting the pod), as well as the addition and subtraction of serialized objects from the (they can be accessed and changed as a Kubernetes volume or downloaded on the fly from the FL Repository in the case of data transformations and models). By default, the inference component accepts data in the form of numerical arrays of any shape and uses a TFLite model to provide lightweight and fast inference. However, it is possible to change the input shape and further details with the use of pluggability.

The inference component is, by default, installed with the rest of the Helm chart. Then it can be accessed through service `fllocaloperationslocal-inferenceapp` on port `50051` according to the specification located in `inference_application/code/proto/basic-inference.proto`.

## Privacy

There are two privacy mechanisms implemented to be used by the FL System. The FL Training Collector can be configured to work with either of them, both or none of them through the use of the training configuration.

### Differential Privacy

The mechanism of [Adaptive Differential Privacy](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/dpfedavg_adaptive.py) modifies the selected strategy by introducing noise to the local model parameters before they are sent by the client. This increases the privacy of the data on the client by obfuscating the information about its distribution. This specific implementation additionally uses adaptive clipping to reduce the balance the influence of multiple clients. The use of this privacy technique may lead to a degradation in the performance of the final model, but introduces little to none additional, computational cost.  

The use of adaptive differential privacy and its specific parameters can be specified in the training configuration under the `privacy_mechanisms` keyword. If we include `dp-adaptive` in this dictionary, we can specify the parameters used by the Flower implementation under the `dp-adaptive` key and configure the training like this:

```json
"privacy-mechanisms":{
  "dp-adaptive":{
    "num_sampled_clients":"1"
  }
}
```
### Homomorphic Encryption

The mechanism of Federated Averaging with Homomorphic Encryption has been implemented from scratch using the [TenSEAL](https://github.com/OpenMined/TenSEAL) library. As Homomorphic Encryption allows for the encryption of numbers such that the decrypted sum of encrypted numbers is the same as the sum of encrypted numbers (and similarly for the subtraction and multiplication). It therefore allows the FL clients to send their encrypted weights, which can then be aggregated and return as the averaged weights in the encrypted form. This ensures that in the event of a malicious server (or a malicious eavesdropper) the privacy of the clients' data remains intact.

The current implementation encrypts the parameters as a CCKS tensor (as implemented in TenSEAL), so if the user would like to generate and serialize new keys and contexts, they should be compatible with this method. 

In order to generate a new set of keys, you can use the file `application/generate_homomorphic_keys.py`. If a new set of keys is generated, the `application/src/custom_clients/hm_keys/public.text` and `application/src/custom_clients/hm_keys/secret.text` files should be appropriately changed (and potentially modified to be a Kubernetes secret).

**Attention**: As an extremely computationally expensive method, it can usually be used only for the simplest of methods and datasets. Therefore it is not recommended in this implementation to use it for models more complicated than a simple Linear Regression.

## Pluggable modules

The trainingapp component suports FL training with the use of Keras and Pytorch libraries out of the box. Similarly, the inferenceapp component supports the inference with the TFLite inferencer. However, it is possible to develop custom components for:
- in the case of trainingapp:
  - FL client
  - FL model
  - FL data loader
  - FL data transformations
- in the case of inferenceapp:
  - gRPC service along with the proto and protocompiled files
  - inferencer
  - model. 

In order to deploy the image with your custom components through the use of Kubernetes volume, change the `custom_setup` field in `values.yaml` to `True`.

For the extended documentation on how to develop pluggable modules based on some examples, please send me an [email](mailto:bogacka@ibspan.waw.pl)

## Prometheus metric monitoring

The Prometheus metrics are available for scraping on the the port `9050` under url `/metrics` on the trainingapp, and on the port `9000` without any additional url path changes in the inferenceapp.

## Authors


- Karolina Bogacka
- Piotr Sowiński
- Jose Antonio Clemente Perez


## Licensing

The FL Local Operations is released under the Apache 2.0 license, as we have internally concluded that we are not "offering the functionality of MongoDB, or modified versions of MongoDB, to third parties as a service". However, potential future commercial adopters should be aware that our project uses MongoDB in order to be able to accurately determine the license most applicable to their projects. 

## Additional documentation

The extended or apprioprately modified documentation will be possible to find [here](https://assist-iot-enablers-documentation.readthedocs.io/en/latest/verticals/federated/fl_local_operations.html).
