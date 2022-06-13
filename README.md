# FL Local Operations


Run `docker-compose -p appv0 --env-file application/.env up --force-recreate --build -d` to run the server and change `USER_INDEX` in `application/.env` file, as well as the `p` argument in the command, to run multiple clients at the same time.
Use FastAPI functionalities to test the API on http://127.0.0.1:9050/docs.
Sample request body for post /job/config/{training_id}:
{"client_type_id": "local1",
 "server_address": "training_collector",
 "optimizer": "adam",
 "eval_metrics": ["MSE", "accuracy"],
 "eval_func": "categorical_crossentropy",
 "num_classes": "10",
 "num_rounds": "50",
 "training_id":"10",
 "model_name": "name",
 "model_version":"10",
 "shape": ["32", "32", "3"],
 "eval_metrics_value": "3.14",
 "config": [{"config_id": "min_effort",
   "batch_size": "64",
   "steps_per_epoch": "32",
   "epochs": "5",
   "learning_rate": "0.001"}]}

Additional changes:
- The data used for training is now taken from the `data` folder by loading the right `x_train.npy`, 
`x_test.npy`, `y_test.npy` and `y_train.npy` files
- The model is now loaded from the local enabler repository and, if it's not there, from the general repository
- The script for starting multiple Local Operations can now be ran as `./start-local.sh 3`, where 3 is a sample number of enablers to start
