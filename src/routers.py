from datetime import datetime
from fastapi import APIRouter, UploadFile, File
import requests
from typing import List
import tempfile

from src.core.models import TrainingRequest
from src.core.testing import testing
from src.core.training import model_train
from src.io.results import PredictionResults, TrainingResults
from src.io.storing import TrainedModel

router = APIRouter()

def inner_train(treq, configuration, csv_filename):
    trained_model, cartid = model_train(treq.trolleyids, treq.start, treq.end, configuration, csv_filename)
    mongoport = configuration.get('mongo_port')
    mongoip = configuration.get('mongo_address')
    mongourl = ('http://' + mongoip + ':' + str(mongoport) + '/ddmodels?identifier=' + str(treq.modelid) +
                '&version=' + str(treq.modelversion))
    result = TrainedModel()#ciao
    result.addModel(trained_model[0])
    result.addRanges(trained_model[2])
    result.addCartID(cartid)
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'wb') as f:
        f.write(result.serialize())
    handler = open(tmp.name, 'rb')
    files = {"file": (tmp.name, handler, "multipart/form-data")}
    resp = requests.post(url=mongourl, files=files)
    return resp.json()



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
    conf = router.configuration
    mongoport = conf.get('mongo_port')
    mongoip = conf.get('mongo_address')
    mongourl = ('http://' + mongoip + ':' + str(mongoport) + '/ddmodels?identifier=' + str(t_req.modelid) +
                '&version=' + str(t_req.modelversion))
    resp = requests.get(url=mongourl)
    tmp = tempfile.NamedTemporaryFile()
    handler = open(tmp.name, 'wb')
    handler.write(resp.content)
    handler.flush()
    reader = open(tmp.name, 'rb')
    fileContent = reader.read()
    trainedmodel = TrainedModel()
    trainedmodel.load(fileContent)
    mdl = trainedmodel.getModel()
    cid = trainedmodel.getCartID()
    rng = trainedmodel.getRanges()
    #todo: ipotesi di prediction effettuata su un solo carrello per volta. Il carrello deve essere presente nella
    # lista di training e il carrello va passato come lista con un solo elemento.
    rul = testing(conf, mdl, cid, t_req.trolleyids[0], t_req.start, t_req.end, t_req.csv_no_influx)
    tr = TrainingResults(rng)
    results = PredictionResults(rul, tr)
    retval = results.getDictionary()
    return retval
