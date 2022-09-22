from http import HTTPStatus

import gridfs
import uvicorn
from config import PORT, HOST, DB_PORT
from fastapi import BackgroundTasks
from fastapi import FastAPI, status, UploadFile, File, Response, HTTPException
from pymongo import MongoClient

import src.local_clients
from application.config import DATABASE_NAME
from datamodels.models import LOTrainingConfiguration, MLModel

app = FastAPI()


# Receive configuration for training job
@app.post("/job/config/{training_id}")
async def receive_updated(training_id, data: LOTrainingConfiguration, background_tasks: BackgroundTasks):
    try:
        if training_id not in src.local_clients.current_jobs:
            src.local_clients.current_jobs[training_id] = 1
        else:
            src.local_clients.current_jobs[training_id] += 1
        background_tasks.add_task(src.local_clients.start_client, training_id=training_id, config=data)
    except Exception as e:
        print("An exception occurred ::", e)
        return 500
    return 200


# Receive  new shared model configuration
@app.post("/model/")
def receive_conf(model: MLModel):
    try:
        client = MongoClient(DATABASE_NAME, DB_PORT)
        db = client.repository
        db.models.insert_one(model.dict(by_alias=True))
    except Exception as e:
        print("An exception occurred ::", e)
        return 500
    return 200


# Receive new model file
@app.put("/model/{model_name}/{model_version}", status_code=status.HTTP_204_NO_CONTENT)
async def update_model(model_name: str, model_version: str, file: UploadFile = File(...)):
    client = MongoClient(DATABASE_NAME, DB_PORT)
    db = client.repository
    db_grid = client.repository_grid
    fs = gridfs.GridFS(db_grid)
    if len(list(db.models.find({'model_name': model_name, 'model_version': model_version}).limit(1))) > 0:
        data = await file.read()
        model_id = fs.put(data, filename=f'model/{model_name}/{model_version}')
        db.models.update_one({'model_name': model_name, 'model_version': model_version}, {"$set": {"model_id": str(model_id)}},
                             upsert=False)
        return Response(status_code=HTTPStatus.NO_CONTENT.value)
    else:
        raise HTTPException(status_code=404, detail="model not found")


# Returns statuses of currently running jobs by returning information
# about the number of model ids and versions being ran
@app.post("/job/status")
def retrieve_status():
    return src.local_clients.current_jobs


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT)
