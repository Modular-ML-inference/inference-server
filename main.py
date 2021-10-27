import uvicorn
from fastapi import FastAPI

from pydloc.models import TrainingConfiguration
from fastapi import BackgroundTasks

from src.local_operations import start_client

app = FastAPI()


# Receive configuration for training job
@app.post("/job/config/{id}")
def receive_updated(id,  data: TrainingConfiguration, background_tasks: BackgroundTasks):
    background_tasks.add_task(start_client, config=data)
    return "Weights Received"


# Receive  new shared model
@app.post("/job/config/{id}/")
def receive_conf(id, model):
    # TODO: Add the model to a database
    return "Received model"


# Receive any required data transformer for job with identified id
@app.post("/job/status/{id}")
def retrieve_status(id):
    return "Receive transformer"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)