# FL Local Operations


Run python main.py to run the server.
Use FastAPI functionalities to test the API on http://127.0.0.1:9000/docs.
Sample request body for post /job/config/{id}:
{
  "model": "cifar10",
  "epochs": 1,
  "rounds": 1,
  "optimizer": "adam",
  "strategy": "string",
  "server_address": "127.0.0.1:8080",
  "batch_size": 32,
  "steps_per_epoch": 1
}