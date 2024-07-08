from datetime import datetime
from fastapi import APIRouter, UploadFile, File, requests
from typing import List

from src.core.models import TrainingRequest
from src.core.training import model_train
from src.io.results import PredictionResults

router = APIRouter()

@router.post("/train")
def train(t_req: TrainingRequest) -> dict:
    conf = router.configuration
    trained_model = model_train(t_req.modelid, t_req.modelversion, t_req.trolleyids, t_req.start, t_req.end, conf)
    mongoport = conf.get('mongo_port')
    mongoip = conf.get('mongo_address')
    mongourl = ('http://' + mongoip + ':' + str(mongoport) + '/ddmodels?identifier=' + str(t_req.modelid) + '&version=' + str(t_req.modelversion))
    #todo: pickle the trained model and flush
    handler = open('modelfitted.keras', 'rb') ##todo: da cambiare
    files = {"file": (handler.name, handler, "multipart/form-data")}
    resp = requests.post(url=mongourl, files=files)
    return resp.json()['success']

@router.get("/train_stupid")
def train(modelid: int, modelversion: int, trolleyids: List[str]) -> dict:
    return {'success': True}

@router.get("/train_service")
def train(modelid: int, modelversion: int, trolleyids: List[str],
          start: datetime, end: datetime, file: UploadFile = File(...)) -> dict:
    #todo call the function making the training
    return {"success": True}

@router.get("/predict")
def postddmodel(modelid: int, modelversion: int, trolleyid: str, start: datetime, end: datetime) -> dict:

    predicted_rul = 0
    std = 0
    #todo call the function making the prediction
    results = PredictionResults()
    #todo: bisogna usare i predictiedresutls per caricare il dizionario giusto
    return results.getDictionary()
