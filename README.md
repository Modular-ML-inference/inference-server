
# FL Local Operations


Run `docker compose -p appv0 up --force-recreate --build -d` to run the server and set `USER_INDEX` to a given value beforehand, as well as the `p` argument in the command to `appv{index}`, to run multiple clients at the same time.
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

Caveat for the kubernetes deployment: in order to properly deploy all containers and all volumes, you have to look into the values.yaml file first and change a couple of local values.