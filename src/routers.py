from datetime import datetime
from fastapi import APIRouter, UploadFile, File, requests
from typing import List

from src.core.models import TrainingRequest
from src.core.training import model_train
from src.io.results import PredictionResults, TrainingResult

router = APIRouter()

def inner_train(treq, configuration, csv_filename):
    trained_model = model_train(treq.modelid, treq.modelversion, treq.trolleyids, treq.start, treq.end,
                                configuration, csv_filename)
    mongoport = configuration.get('mongo_port')
    mongoip = configuration.get('mongo_address')
    mongourl = ('http://' + mongoip + ':' + str(mongoport) + '/ddmodels?identifier=' + str(treq.modelid) +
                '&version=' + str(treq.modelversion))
    #todo: pickle the trained model and flush
    handler = open('modelfitted.keras', 'rb') ##todo: da cambiare
    files = {"file": (handler.name, handler, "multipart/form-data")}
    resp = requests.post(url=mongourl, files=files)
    return resp.json()['success']



@router.post("/train")
def train(t_req: TrainingRequest) -> dict:
    conf = router.configuration
    csv_filename = None
    retval = inner_train(t_req, conf, csv_filename)
    return retval

@router.post("/train_service")
def train(t_req: TrainingRequest) -> dict:
    conf = router.configuration
    csv_filename = t_req.csv_no_influx
    retval = inner_train(t_req, conf, csv_filename)
    return retval

@router.post("/predict")
def postddmodel(t_req: TrainingRequest) -> dict:

    predicted_rul = 100
    #todo call the function making the prediction
    tr = TrainingResult()
    tr.add('less-than-1-hour', 105, 120, 140)
    results = PredictionResults(predicted_rul, tr)
    #todo: bisogna usare i predictiedresutls per caricare il dizionario giusto
    return results.getDictionary()
