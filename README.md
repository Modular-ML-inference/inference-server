# FL Local Operations


Run python main.py to run the server.
Use FastAPI functionalities to test the API on http://127.0.0.1:9000/docs.
Sample request body for post /job/config/{id}:
{"client_type_id": "local1",
 "server_address": "127.0.0.1",
 "optimizer": "adam",
 "eval_metrics": ["MSE"],
 "eval_func": "Huber",
 "num_classes": "2",
 "shape": ["64", "32", "2"],
 "model_id": "base",
 "config": [{"config_id": "min_effort",
   "batch_size": "64",
   "steps_per_epoch": "32",
   "epochs": "50",
   "learning_rate": "0.001"}]}