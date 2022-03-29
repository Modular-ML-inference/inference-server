# FL Local Operations


Run python main.py to run the server.
Use FastAPI functionalities to test the API on http://127.0.0.1:9050/docs.
Sample request body for post /job/config/{id}:
{"client_type_id": "local1",
 "server_address": "training_collector",
 "optimizer": "adam",
 "eval_metrics": ["MSE"],
 "eval_func": "Huber",
 "num_classes": "10",
 "model_id":"10",
 "model_version":"10",
 "shape": ["32", "32", "3"],
 "config": [{"config_id": "min_effort",
   "batch_size": "64",
   "steps_per_epoch": "32",
   "epochs": "5",
   "learning_rate": "0.001"}]}