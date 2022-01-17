import uvicorn
from fastapi import BackgroundTasks
from fastapi import FastAPI
from pymongo import MongoClient

from config import PORT, HOST, DB_PORT
from pydloc.models import LOTrainingConfiguration, MLAlgorithm
from src.local_clients import start_client

app = FastAPI()


# Receive configuration for training job
@app.post("/job/config/{id}")
def receive_updated(id,  data: LOTrainingConfiguration, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(start_client, id=id, config=data)
    except Exception as e:
        print("An exception occurred ::", e)
        return 500
    return 200


# Receive  new shared model
@app.post("/model/")
def receive_conf(model: MLAlgorithm):
    try:
        client = MongoClient(port=DB_PORT)
        db = client.local
        db.models.insert_one(model.dict(by_alias=True))
    except Exception as e:
        print("An exception occurred ::", e)
        return 500
    return 200


# Receive any required data transformer for job with identified id
@app.post("/job/status/{id}")
def retrieve_status(id):
    return "Receive transformer"


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)