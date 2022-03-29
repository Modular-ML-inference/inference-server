import gridfs
import uvicorn
from http import HTTPStatus
from fastapi import BackgroundTasks
from fastapi import FastAPI, status, UploadFile, File, Response, HTTPException
from pymongo import MongoClient
from config import PORT, HOST, DB_PORT
from pydloc.models import LOTrainingConfiguration, MLModel
from src.local_clients import start_client

app = FastAPI()


# Receive configuration for training job
@app.post("/job/config/{id}")
def receive_updated(id, data: LOTrainingConfiguration, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(start_client, id=id, config=data)
    except Exception as e:
        print("An exception occurred ::", e)
        return 500
    return 200


# Receive  new shared model configuration
@app.post("/model/")
def receive_conf(model: MLModel):
    try:
        client = MongoClient('db', DB_PORT)
        db = client.local
        db.models.insert_one(model.dict(by_alias=True))
    except Exception as e:
        print("An exception occurred ::", e)
        return 500
    return 200


# Receive new model file
@app.put("/model/{id}/{version}", status_code=status.HTTP_204_NO_CONTENT)
async def update_model(id: int, version: int, file: UploadFile = File(...)):
    client = MongoClient('db', DB_PORT)
    db = client.repository
    db_grid = client.repository_grid
    fs = gridfs.GridFS(db_grid)
    if len(list(db.models.find({'id': id, 'version': version}).limit(1))) > 0:
        data = await file.read()
        model_id = fs.put(data, filename=f'model/{id}/{version}')
        db.models.update_one({'id': id, 'version': version}, {"$set": {"model_id": str(model_id)}},
                             upsert=False)
        return Response(status_code=HTTPStatus.NO_CONTENT.value)
    else:
        raise HTTPException(status_code=404, detail="model not found")


# Receive any required data transformer for job with identified id
@app.post("/job/status/{id}")
def retrieve_status(id):
    return "Receive transformer"


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
