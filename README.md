
# FL Local Operations


Run `docker compose -p appv0 up --force-recreate --build -d` to run the server and set `USER_INDEX` to a given value beforehand, as well as the `p` argument in the command to `appv{index}`, to run multiple clients at the same time.
You should also set up beforehand the value of `FL_LOCAL_OP_DATA_FOLDER`.

You can also run `USER_INDEX=5 FL_LOCAL_OP_DATA_FOLDER="./data" docker compose up --force-recreate --build -d` to setup those necessary flags yourself without relying on a script.
Use FastAPI functionalities to test the API on http://127.0.0.1:9050/docs.

Additional changes:
- The data used for training is now taken from the `data` folder by loading the right `x_train.npy`, 
`x_test.npy`, `y_test.npy` and `y_train.npy` files
- The model is now loaded from the local enabler repository and, if it's not there, from the general repository
- The script for starting multiple Local Operations can now be ran as `./start-local.sh 3`, where 3 is a sample number of enablers to start


Configuration for Twotronics demo in kubernetes:
Sample request body for post /job/config/{training_id}:
```json
{
  "client_type_id": "local1",
  "server_address": "fltrainingcollectorlocal-trainingmain-svc2",
  "eval_metrics": [
    "precision", "recall"
  ],
  "eval_func": "categorical_crossentropy",
  "num_classes": 10,
  "num_rounds": 15,
  "shape": [
    32, 32, 3
  ],
  "training_id": "10",
  "model_name": "twotronics",
  "model_version": "demo",
  "config": [
    {"config_id": "min_effort",
   "batch_size": "64",
   "steps_per_epoch": "32",
   "epochs": "1",
   "learning_rate": "0.001"}
  ],
  "optimizer_config": {
    "optimizer": "sgd",
    "lr": "0.005",
    "momentum": "0.9",
    "weight_decay": "0.0005"
  },
  "scheduler_config": {
    "scheduler": "steplr",
    "step_size": "3",
    "gamma": "0.1"
  },
  "warmup_config": {
    "scheduler": "lambdalr",
    "warmup_iters": "1000",
    "warmup_epochs": "1",
    "warmup_factor": "0.001",
    "scheduler_conf": {
      "scheduler": "lambdalr"
    }
  },
  "privacy-mechanisms":{"dp-adaptive":{"num_sampled_clients":"1"}},
  "eval_metrics_value": "0"
}
```

This enabler can use homomorphic encryption for communication. In order to generate a new set of keys, run the file `application/generate_homomorphic_keys.py`

# Kubernetes configuration

In order to properly set up the enabler with the use of Helm charts, first you have to set up the appropriate configuration. For this purposes, the `fllocalops-config-map.yaml` is included in this repository. This is a ConfigMap containing information that may be specific to this deployment that the application must be able to access.After performing appropriate modifications, run `kubectl apply -f fllocalops-config-map.yaml` to create the ConfigMap.

Before running the helm chart, you also have to set up a new (in particular, at this moment I have set up this persistent volume to not be reusable) PV. To do so, first go to `local-pv.yaml` and make sure that path placed there is an absolute path to the volume with your training data in your repository. Then, run `kubectl apply -f local-pv.yaml` to create the persistent volume. You also have to run `kubectl cp ./data <podname>:/data` in order to copy data to persistent volume (I will try to figure out a better solution using rsync for later use).

Finally, run `helm install fllocaloperationslocal fllocaloperations` in order to properly install the release using Helm charts.

You can later use `kubectl port-forward <podname> <hostport>:9050` to forward port to your localhost and easily set up local configuration on `127.0.0.1:<hostport>/docs`.

A sample configuration that can be input on /docs and used to test the keras builder (currently on mock downloaded data) is:
```json
{"client_type_id": "local1",
  "server_address": "trainingcollectorlocal-trainingmain-svc2",
  "eval_metrics": [
    "precision", "recall"
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
  "privacy-mechanisms":{},
  "eval_metrics_value": "0"}
```
Conversely, a sample config for twotronics is:
```json
{
  "strategy": "avg",
  "model_name": "twotronics",
  "model_version": "demo",
  "adapt_config": "custom",
  "server_conf": {
    "num_rounds": 3
  },
  "strategy_conf": {
    "min_fit_clients" : "1",
    "min_available_clients": "1",
    "min_evaluate_clients": "1"
  },
"privacy-mechanisms":{
},
  "client_conf": [
    {
      "config_id" : "min_effort",
      "batch_size": "64",
      "steps_per_epoch" : "32",
      "epochs" : "5",
      "learning_rate" : "0.001"
    }
  ],
  "configuration_id": "10"
}
```