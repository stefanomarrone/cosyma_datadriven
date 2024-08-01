import ast
import functools
import os
import random
import json
import requests
import itertools
import polars as pl
import pandas as pd
import numpy as np
import time as ti
import time as t
import tensorflow as tf
import pyarrow
import warnings
import pytz
import influxdb_client
import traceback
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta
from configparser import ConfigParser
from keras import Sequential, Input
from keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from keras.models import load_model, save_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import scipy.stats as stats

def upload_path(config_path):
    config = ConfigParser()
    
    # Leggi il file esistente
    config.read(config_path)
    # Leggi i path da ..config_file.ini
    modelPath = config.get('connection', 'ModelPath')
    
    return modelPath



def update_testing_dates(file_path, test_start, test_stop, serial_number):
    config = ConfigParser()
    
    # Leggi il file esistente
    config.read(file_path)
    
    # Se la sezione 'connection' non esiste, creala
    if 'connection' not in config.sections():
        config.add_section('connection')
    
    # Aggiorna i valori di TrainStart e TrainStop
    config['connection']['TestStart'] = test_start
    config['connection']['TestStop'] = test_stop
    config['connection']['serialNumber'] = serial_number
    
    # Scrivi le modifiche nel file
    with open(file_path, 'w') as configfile:
        config.write(configfile)



def update_testing_ini(file_path, model_path):
    config = ConfigParser()
    
    # Leggi il file esistente
    config.read(file_path)
    
    # Se la sezione 'connection' non esiste, creala
    if 'connection' not in config.sections():
        config.add_section('connection')
    
    # Aggiorna i valori di ModelPath
    config['connection']['ModelPath'] = model_path
    
    # Scrivi le modifiche nel file
    with open(file_path, 'w') as configfile:
        config.write(configfile)



def update_training_ini(file_path, train_start, train_stop):
    config = ConfigParser()
    
    # Leggi il file esistente
    config.read(file_path)
    
    # Se la sezione 'connection' non esiste, creala
    if 'connection' not in config.sections():
        config.add_section('connection')
    
    # Aggiorna i valori di TrainStart e TrainStop
    config['connection']['TrainStart'] = train_start
    config['connection']['TrainStop'] = train_stop
    
    # Scrivi le modifiche nel file
    with open(file_path, 'w') as configfile:
        config.write(configfile)



def read_ranges(file_path):
    # Inizializzare il parser
    config = ConfigParser()

    # Leggere il file ranges.ini
    config.read(file_path)

    # Liste per memorizzare i valori
    more_than_3_hours = []
    between_3_hours_and_1_hour = []
    less_than_1_hour = []

    # Funzione per estrarre i valori da una sezione e aggiungerli alla lista appropriata
    def extract_values(section, target_list):
        target_list.append(float(config[section]['ninety']))
        target_list.append(float(config[section]['ninety-five']))
        target_list.append(float(config[section]['ninety-nine']))

    # Estrarre i valori e aggiungerli alle liste
    extract_values('more than 3 hours', more_than_3_hours)
    extract_values('between 3 hours and 1 hour', between_3_hours_and_1_hour)
    extract_values('less than 1 hour', less_than_1_hour)

    return more_than_3_hours, between_3_hours_and_1_hour, less_than_1_hour



def create_ranges_diff_ini(data):
    # Creare un oggetto ConfigParser
    config = ConfigParser()

    # Definire le chiavi per ciascuna sezione
    sections = ["more than 3 hours", "between 3 hours and 1 hour", "less than 1 hour"]
    keys = {0.1: "ninety", 0.05: "ninety-five", 0.01: "ninety-nine"}

    # Popolare il file di configurazione con le differenze
    for section, dic in zip(sections, data):
        config[section] = {keys[key]: str(dic[key][1] - dic[key][0]) for key in dic}

    # Scrivere il file ranges.ini
    rangesName = './Testing.ini'
    with open(rangesName, 'w') as configfile:
        config.write(configfile)



def create_ranges_ini(data, path):
    # Creare un oggetto ConfigParser
    config = ConfigParser()

    # Definire le chiavi per ciascuna sezione
    sections = ["more than 3 hours", "between 3 hours and 1 hour", "less than 1 hour"]
    keys = {0.1: "ninety", 0.05: "ninety-five", 0.01: "ninety-nine"}

    # Popolare il file di configurazione
    for section, dic in zip(sections, data):
        config[section] = {keys[key]: str(dic[key]) for key in dic}

    # Scrivere il file ranges.ini
    rangesName = 'ranges.ini'
    writePth = os.path.join(path, rangesName)
    with open(writePth, 'w') as configfile:
        config.write(configfile)



def update_config_dates(training_config_path, main_config_path):
    # Crea i parser per entrambi i file di configurazione
    training_config = ConfigParser()
    main_config = ConfigParser()

    # Leggi i file di configurazione
    training_config.read(training_config_path)
    main_config.read(main_config_path)

    # Leggi le date da Training.ini
    train_start = training_config.get('connection', 'TrainStart')
    train_stop = training_config.get('connection', 'TrainStop')

    # Aggiorna le date in config.ini
    main_config.set('connection', 'TrainStart', train_start)
    main_config.set('connection', 'TrainStop', train_stop)

    # Scrivi le modifiche nel file config.ini
    with open(main_config_path, 'w') as configfile:
        main_config.write(configfile)



def update_config_dates_test(testing_config_path, main_config_path, serialN):
    # Crea i parser per entrambi i file di configurazione
    testing_config = ConfigParser()
    main_config = ConfigParser()

    # Leggi i file di configurazione
    testing_config.read(testing_config_path)
    main_config.read(main_config_path)

    # Leggi le date da Training.ini
    test_start = testing_config.get('connection', 'TestStart')
    test_stop = testing_config.get('connection', 'TestStop')
    #serialN = testing_config.get('connection', 'serialNumber')
    
    # Aggiorna le date in config.ini
    main_config.set('connection', 'TestStart', test_start)
    main_config.set('connection', 'TestStop', test_stop)
    main_config.set('connection', 'serialNumber', serialN)
    
    # Scrivi le modifiche nel file config.ini
    with open(main_config_path, 'w') as configfile:
        main_config.write(configfile)



def createFld(motherPath, folderName):
    path = os.path.join(motherPath,folderName)
    if not os.path.exists(path): 
        os.mkdir(path)
    return path


def readingCSVDB(filename):
    data = pl.scan_csv(filename, separator = ",", comment_prefix = "#", has_header = True, try_parse_dates = True, dtypes = {'_value': pl.Float64})
    return data
    
    
def readingCSV(name, folder):
    path = os.path.join(folder,name)
    data = pl.scan_csv(path, separator = ",", comment_prefix = "#", has_header = True, try_parse_dates = True, dtypes = {'life': pl.Int64})
    return data











########## SETTING - END ##########











########## PREPROCESSING ##########






##### DOWNLOAD #####



def readingWithInfluxDB(bucket, org, token, url, start, stop):
    
    #### CONNECTION ####
    client = influxdb_client.InfluxDBClient(
        url=url,
        token=token,
        org=org
    )
    
    #### QUERY SCRIPT ####
    query_api = client.query_api()
    query = 'from(bucket:"cosymaDB")\
    |> range(start: ' + start + ', stop: ' + stop + ')'
    #|> range(start: -10m, stop: -10m)'
    #2021-01-01T00:00:00Z, stop: 2021-01-01T12:00:00Z
    #### EXECUTION OF QUERY ####
    result = query_api.query(org=org, query=query)
    
    
    #### READING OF RESULTS ####
    results = []
    for table in result:
        for record in table.records:
            results.append((record.get_field(), record.get_value()))
    

    #### CREATION DB ####
    # Definisci i nomi delle colonne
    nomi_colonne = ['table', '_start', '_stop', '_time', '_value', '_field', '_measurement', 'ns'] #['result', 'table', '_start', '_stop', '_time', '_value', '_field', '_measurement', 'ns']
    diction = dict()
    for name in nomi_colonne:
        diction[name] = []
     
    # Crea un DataFrame lazy con colonne vuote
    #print(diction)    
    df = pl.LazyFrame(diction, schema={"table": pl.Int64, "_start": pl.Datetime, "_stop": pl.Datetime, "_time": pl.Datetime, "_value": pl.Float64, "_field": pl.String, "_measurement": pl.String, "ns": pl.String})
    
    for table in result:
        for record in table.records:
            riga = list()
            for column, value in zip(table.columns, record.values):
                if (value == "_start") | (value == "_stop") | (value == "_time"):
                    record[value] = str(record[value]) #record[value].map(lambda x: str(datetime.strptime(x.split('+')[0], "%Y-%m-%d %H:%M:%S.%f")))
                    parte_da_rimuovere = "+00:00"
                    record[value] = record[value].replace(parte_da_rimuovere, "")
                    formato_data = "%Y-%m-%d %H:%M:%S.%f"
                    
                    # Analizza la stringa della data nel formato specificato
                    #### resolving .%f issue ####
                    try:
                        record[value] = datetime.strptime(record[value], formato_data)

                    # If that doesn't work, we add ".4" to the end of stDate
                    # (You can change this to ".0")
                    # We then retry to convert stDate into datetime format                                   
                    except:
                        record[value] = record[value] + ".0"
                        record[value] = datetime.strptime(record[value], formato_data)
                    
                    # Analizza la stringa della data nel formato specificato
                    #record[value] = datetime.strptime(record[value], formato_data)  #### module 'datetime' has no attribute 'strptime'   np.todatetime(), devi sempre mettere il formato
                riga.append(record[value])
            nuova_riga = pl.LazyFrame({'table': riga[1], "_start": riga[2], "_stop": riga[3], "_time": riga[4], "_value": riga[5], "_field": riga[6], "_measurement": riga[7], "ns": riga[8]})
            df = pl.concat([df, nuova_riga])
            
    #print(df.collect())
    #### TUTTO OK!
    return df


def savePandasDataFrame(data, name, folder):
    if not os.path.exists(folder): 
        os.mkdir(folder)
    path = os.path.join(folder, name)
    data.to_csv(path)
    return 0


def savePolarsLazyFrameCSV(data, name, folder):
    if not os.path.exists(folder): 
        os.mkdir(folder)
    path = os.path.join(folder, name)
    data.collect().write_csv(path)
    return 0


def savePolarsDataFrameCSV(data, name, folder):
    if not os.path.exists(folder): 
        os.mkdir(folder)
    path = os.path.join(folder, name)
    data.write_csv(path)
    return 0


def onlyMeasRead(id, dataset):
    colNm = ['time','event','descr','value']
    # date conversion (maybe the real problem, naah)
    datetimeStr = str(dataset.iloc[id,0])
    formato_data = '%Y-%m-%d %H:%M:%S.%f'
    #######datetimeObject = datetime.strptime(datetimeStr, '%Y-%m-%d %H:%M:%S.%f') FORMAT ISSUE
    
    #### RESOLVING FORMAT ISSUE ####
    try:
        datetimeObject = datetime.strptime(datetimeStr, formato_data)

        # If that doesn't work, we add ".4" to the end of stDate
        # (You can change this to ".0")
        # We then retry to convert stDate into datetime format                                   
    except:
        datetimeStr = datetimeStr + ".0"
        datetimeObject = datetime.strptime(datetimeStr, formato_data)
    #### RESOLVED ####
    
    firstRow = np.array([[datetimeObject], [dataset.iloc[id,1]], [dataset.iloc[id,2]], [dataset.iloc[id,3]]]).T
    subSet = dataset.iloc[(id + 1):,:]
    plData = pl.LazyFrame(subSet)
    onlMsQry = plData.select(pl.col('*')).filter(pl.col('list of ev') == 'Meas')
    result = onlMsQry.collect().to_numpy()

    npResult = np.concatenate([firstRow,result], axis = 0)
    result = pd.DataFrame(npResult, columns=colNm)
    #result = pl.from_numpy(npResult) if we want to use polars
    
    return result



##### DOWNLOAD _ END #####






##### FIRST PREPROCESSING #####



def SampleViewCreate(data):
    
    ### Ordiniamo i dati nel tempo (in ordine decrescente - dal pi� recente al pi� vecchio)
    data = data.select(pl.col('*')).sort(by = '_time', descending = True)

    ### Definizione della vista combinata: Fail&Meas View
    # step 1: add representative fault event col
    colToAdd1 = pl.lit('EV_FAULT')
    colToAdd2 = pl.lit('MEASUREMENT')
    dataComb = data.with_columns(new_column1 = colToAdd1)
    dataComb = dataComb.with_columns(new_column2 = colToAdd2)
    dataComb = dataComb.rename({'new_column1' : 'faultEv'})
    dataComb = dataComb.rename({'new_column2' : 'measEv'})
    # step 2: select rows with event informations
    queryCombEvent = (dataComb.select(pl.col('*')).filter( (pl.col('_measurement').str.contains(pl.col('faultEv'))) | (pl.col('_measurement').str.contains(pl.col('measEv'))) )).select(pl.col('_value'), pl.col('_time'), pl.col('ns'), pl.col('_measurement'))
    # step 3: remove unnecessary col
    queryCombEvent = queryCombEvent.select(pl.col(name) for name in queryCombEvent.columns if name != 'faultEv')
    queryCombEvent = queryCombEvent.select(pl.col(name) for name in queryCombEvent.columns if name != 'measEv')
    combView = queryCombEvent    
    #print(combView.filter(pl.col('_value') == 1.0).collect())
    ### restruct cartPath to combine fail events with meas events
    cartNames = combView.select(pl.col('ns')).collect().to_series()
    cartNames = pd.Series(cartNames)
    # for each path, remove non necessary info
    newName = []
    for name in cartNames:
        charIsDot = 0
        element = 0
        while (element in range(0, len(name))) & (charIsDot == 0):
            if name[element] == ".":
                firstDotIndex = element
                charIsDot = 1
            element = element + 1
        newName.append(name[:firstDotIndex])

    newCol = pd.Series(newName)
    newCartNames = pd.concat([cartNames,newCol], axis = 1)
    newCartNames.columns = ['long Id', 'short Id']

    colToAdd1 = pl.lit(pl.Series('long Id', cartNames))
    colToAdd2 = pl.lit(pl.Series('short Id', newCol))
    combView = combView.with_columns(new_col1 = colToAdd1)
    combView = combView.with_columns(new_col2 = colToAdd2)
    combView = combView.rename({'new_col1' : 'long Id'})
    combView = combView.rename({'new_col2' : 'short Id'})
    #littleQuery = combView.filter( (pl.col('long Id').str.contains('IFC0000950_2023410028')) & (pl.col('_measurement').str.contains('FAULT')) )
    
    ### create two new col (type of event col and descr event col)
    # step 1: extract _meas col
    measQuery = combView.select(pl.col('_measurement'))
    measCol = measQuery.collect().to_series()
    measCol = pd.Series(measCol)
    #step 2: create new col
    desEv = []
    typEv = []
    for event in range(0, len(measCol)):
        if 'EV_FAULT_' in measCol.iloc[event] :
            desName = measCol.iloc[event].replace("EV_FAULT_","")
            desEv.append(desName)
            typEv.append('Fault')
        else:
            desName = measCol.iloc[event].replace("_MEASUREMENT","")
            typEv.append('Meas')
            desEv.append(desName)
    #step 3: add new col in pl.data and remove old meas col
    #adding
    colToAdd1 = pl.lit(pl.Series('event', typEv))
    colToAdd2 = pl.lit(pl.Series('type', desEv))
    combView = combView.with_columns(new_col1 = colToAdd1)
    combView = combView.with_columns(new_col2 = colToAdd2)
    combView = combView.rename({'new_col1' : 'event'})
    combView = combView.rename({'new_col2' : 'type'})
    #removing
    combView = combView.select(pl.col(name) for name in combView.columns if name != '_measurement')
    
    ### remove not important cols
    combView = combView.select(pl.col(name) for name in combView.columns if name != 'ns')
    combView = combView.select(pl.col(name) for name in combView.columns if name != 'long Id')
    
    ### group_by cart name
    cartView = combView.group_by(pl.col('short Id'), maintain_order = True).agg(
        pl.col('_time').alias('list of times'),
        pl.col('event').alias('list of ev'),
        pl.col('type').alias('list of des'),
        pl.col('_value').alias('list of val')
        )
    pandData = cartView.collect().to_pandas()
    
    ### Organize data like the form: meas,meas,meas,...,meas,fault
    # for each cart I can read the data, and when I meet a fault event,
    # I can save the index and print meas info from index to the end.

    lenData = pandData.shape[0]
    frames = []
    for cartID in range(lenData):
        #cartSet creating
        cartSet = pandData.iloc[cartID,1:]
        #four col to create
        firstCol = pd.Series(cartSet.iloc[0], name = 'list of times')
        secondCol = pd.Series(cartSet.iloc[1], name = 'list of ev')
        thirdCol = pd.Series(cartSet.iloc[2], name = 'list of des')
        fourthCol = pd.Series(cartSet.iloc[3], name = 'list of val')
        cols = [firstCol, secondCol, thirdCol, fourthCol]
        cartSet = pd.concat(cols, axis = 1)
        lnCrtData = cartSet.iloc[:,1].shape[0]

        
        ##cosa fare: parto in un for di lunghezza lenCartDataset
            # al suo interno leggo la lista di eventi. 
            # Quando trovo un evento fail, devo eseguire un programmino che chiamo
            # onlymeasread(), in output mi da un pandas df, che concateno con gli altri
            # suoi simili.
        evLst = cartSet.iloc[:,1]
        evID = 0
        cartFrames = []

        while evID in range(lnCrtData):
            #searching right index
            if evLst.iloc[evID] == 'Fault':
                svID = evID
                data = onlyMeasRead(svID, cartSet)
                cartIDx = pd.Series((pandData.iloc[cartID,0] for a in range(data.shape[0])), name = 'cart Id')
                merging = [cartIDx, data]
                merged = pd.concat(merging, axis = 1)
                cartFrames.append(merged)
            evID = evID + 1
        if len(cartFrames) == 1:
            cartData = cartFrames[0]
        else:
            if cartFrames != []:
                cartData = pd.concat(cartFrames, axis = 0)
            else:
                cartData = []
        if type(cartData) != type([]):
            #print("ok")
            frames.append(cartData)

    fullCartData = pd.concat(frames)
    return fullCartData


def SampleViewCreate_test(data):
    
    ### Ordiniamo i dati nel tempo (in ordine decrescente - dal pi� recente al pi� vecchio)
    data = data.select(pl.col('*')).sort(by = '_time', descending = True)

    ### Definizione della vista combinata: Fail&Meas View
    # step 1: add representative fault event col
    colToAdd1 = pl.lit('EV_FAULT')
    colToAdd2 = pl.lit('MEASUREMENT')
    dataComb = data.with_columns(new_column1 = colToAdd1)
    dataComb = dataComb.with_columns(new_column2 = colToAdd2)
    dataComb = dataComb.rename({'new_column1' : 'faultEv'})
    dataComb = dataComb.rename({'new_column2' : 'measEv'})
    # step 2: select rows with event informations
    queryCombEvent = (dataComb.select(pl.col('*')).filter( (pl.col('_measurement').str.contains(pl.col('faultEv'))) | (pl.col('_measurement').str.contains(pl.col('measEv'))) )).select(pl.col('_value'), pl.col('_time'), pl.col('ns'), pl.col('_measurement'))
    # step 3: remove unnecessary col
    queryCombEvent = queryCombEvent.select(pl.col(name) for name in queryCombEvent.columns if name != 'faultEv')
    queryCombEvent = queryCombEvent.select(pl.col(name) for name in queryCombEvent.columns if name != 'measEv')
    combView = queryCombEvent    
    #print(combView.filter(pl.col('_value') == 1.0).collect())
    ### restruct cartPath to combine fail events with meas events
    cartNames = combView.select(pl.col('ns')).collect().to_series()
    cartNames = pd.Series(cartNames)
    # for each path, remove non necessary info
    newName = []
    for name in cartNames:
        charIsDot = 0
        element = 0
        while (element in range(0, len(name))) & (charIsDot == 0):
            if name[element] == ".":
                firstDotIndex = element
                charIsDot = 1
            element = element + 1
        newName.append(name[:firstDotIndex])

    newCol = pd.Series(newName)
    newCartNames = pd.concat([cartNames,newCol], axis = 1)
    newCartNames.columns = ['long Id', 'short Id']

    colToAdd1 = pl.lit(pl.Series('long Id', cartNames))
    colToAdd2 = pl.lit(pl.Series('short Id', newCol))
    combView = combView.with_columns(new_col1 = colToAdd1)
    combView = combView.with_columns(new_col2 = colToAdd2)
    combView = combView.rename({'new_col1' : 'long Id'})
    combView = combView.rename({'new_col2' : 'short Id'})
    #littleQuery = combView.filter( (pl.col('long Id').str.contains('IFC0000950_2023410028')) & (pl.col('_measurement').str.contains('FAULT')) )

    ### create two new col (type of event col and descr event col)
    # step 1: extract _meas col
    measQuery = combView.select(pl.col('_measurement'))
    measCol = measQuery.collect().to_series()
    measCol = pd.Series(measCol)
    #step 2: create new col
    desEv = []
    typEv = []
    for event in range(0, len(measCol)):
        if 'EV_FAULT_' in measCol.iloc[event] :
            desName = measCol.iloc[event].replace("EV_FAULT_","")
            desEv.append(desName)
            typEv.append('Fault')
        else:
            desName = measCol.iloc[event].replace("_MEASUREMENT","")
            typEv.append('Meas')
            desEv.append(desName)
    #step 3: add new col in pl.data and remove old meas col
    #adding
    colToAdd1 = pl.lit(pl.Series('event', typEv))
    colToAdd2 = pl.lit(pl.Series('type', desEv))
    combView = combView.with_columns(new_col1 = colToAdd1)
    combView = combView.with_columns(new_col2 = colToAdd2)
    combView = combView.rename({'new_col1' : 'event'})
    combView = combView.rename({'new_col2' : 'type'})
    #removing
    combView = combView.select(pl.col(name) for name in combView.columns if name != '_measurement')
    
    ### remove not important cols
    combView = combView.select(pl.col(name) for name in combView.columns if name != 'ns')
    combView = combView.select(pl.col(name) for name in combView.columns if name != 'long Id')
    
    ### group_by cart name
    cartView = combView.group_by(pl.col('short Id'), maintain_order = True).agg(
        pl.col('_time').alias('list of times'),
        pl.col('event').alias('list of ev'),
        pl.col('type').alias('list of des'),
        pl.col('_value').alias('list of val')
        )
    pandData = cartView.collect().to_pandas()

    ### Organize data like the form: meas,meas,meas,...,meas,fault
    # for each cart I can read the data, and when I meet a fault event,
    # I can save the index and print meas info from index to the end.

    lenData = pandData.shape[0]
    frames = []
    for cartID in range(lenData):
        #cartSet creating
        cartSet = pandData.iloc[cartID,1:]
        #four col to create
        firstCol = pd.Series(cartSet.iloc[0], name = 'list of times')
        secondCol = pd.Series(cartSet.iloc[1], name = 'list of ev')
        thirdCol = pd.Series(cartSet.iloc[2], name = 'list of des')
        fourthCol = pd.Series(cartSet.iloc[3], name = 'list of val')
        cols = [firstCol, secondCol, thirdCol, fourthCol]
        cartSet = pd.concat(cols, axis = 1)
        lnCrtData = cartSet.iloc[:,1].shape[0]

        ##cosa fare: parto in un for di lunghezza lenCartDataset
            # al suo interno leggo la lista di eventi. 
            # Quando trovo un evento fail, devo eseguire un programmino che chiamo
            # onlymeasread(), in output mi da un pandas df, che concateno con gli altri
            # suoi simili.
        evLst = cartSet.iloc[:,1]
        evID = 0
        cartFrames = []
        flag = False
        while evID in range(lnCrtData):
            #searching right index
            if evLst.iloc[evID] == 'Fault':
                flag = True
                svID = evID
                data = onlyMeasRead(svID, cartSet)
                cartIDx = pd.Series((pandData.iloc[cartID,0] for a in range(data.shape[0])), name = 'cart Id')
                merging = [cartIDx, data]
                merged = pd.concat(merging, axis = 1)
                cartFrames.append(merged)
            evID = evID + 1
        if len(cartFrames) == 1:
            cartData = cartFrames[0]
        else:
            if cartFrames != []:
                cartData = pd.concat(cartFrames, axis = 0)
            else:
                cartData = []
        if type(cartData) != type([]):
            #print("ok")
            frames.append(cartData)
            
        if flag == False:
            cartSet.columns = ['time', 'event', 'descr', 'value']
            cartIDx = pd.Series((pandData.iloc[cartID,0] for a in range(cartSet.shape[0])), name = 'cart Id')
            merging = [cartIDx, cartSet]
            merged = pd.concat(merging, axis = 1)
            frames.append(merged)
    fullCartData = pd.concat(frames)
    return fullCartData



def modify_values(data):
    modified_data = data.with_columns([
        pl.when(pl.col('event') == 'Meas').then(0.0).otherwise(pl.col('Fault')).alias('Fault'),
        pl.when(pl.col('event') == 'Fault').then(0.0).otherwise(pl.col('Meas')).alias('Meas')
    ])
    return modified_data


def MeasFaultSplit(data):
    data = data.with_columns([pl.col('value').alias('Meas'), pl.col('value').alias('Fault')]).select(pl.col('cart Id'), pl.col('time'), pl.col('descr'), pl.col('Meas'), pl.col('Fault'), pl.col('event')) 
    #print(data.collect())
    modified_data = modify_values(data)
    data = modified_data.select(pl.col('cart Id'), pl.col('time'), pl.col('descr'), pl.col('Meas'), pl.col('Fault'))
    #print(modified_data.collect())
    return data


def Preprocessing(data):
    # reading path
    #motherPath = r"C:\Users\Ciro\Desktop\core"
    #readingFileName = "Database.csv" # this is the old dataset: "influx.dataORIGINAL.csv"
    #eventCSVFileName = "Sample View.csv"
    #savingCSVPath = os.path.join(motherPath,eventCSVFileName)
    #readingPath = os.path.join(motherPath,readingFileName)
    
    # reading as polars lazyframe object (faster read)
    lazyData = data #pl.scan_csv(readingPath, separator = ",", comment_prefix = "#", has_header = True, try_parse_dates = True, dtypes = {'_value': pl.Float64})
    #print(lazyData.collect())
    
    ## Remove null cols and null rows
    #row
    lazyData = lazyData.filter(~pl.all_horizontal(pl.all().is_null()))
    #col
    nullCountQuery = lazyData.select(pl.col(name).null_count() for name in lazyData.collect_schema().names())
    data = nullCountQuery.collect().to_pandas()
    shape = lazyData.collect().shape[0]
    colName = [name for name in data.columns if data[name].iloc[0] != shape]
    lazyData = lazyData.select(pl.col(name) for name in colName)
    
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = SampleViewCreate(lazyData)
    
    result = pl.LazyFrame(results)
    results = MeasFaultSplit(result)
    ### saving into csv format
    #results.to_csv(savingCSVPath)
    #print('work done.')
    return results



def Preprocessing_test(data):
    
    lazyData = data 
    
    #row
    lazyData = lazyData.filter(~pl.all_horizontal(pl.all().is_null()))
    #col
    nullCountQuery = lazyData.select(pl.col(name).null_count() for name in lazyData.columns)
    data = nullCountQuery.collect().to_pandas()
    shape = lazyData.collect().shape[0]
    colName = [name for name in data.columns if data[name].iloc[0] != shape]
    lazyData = lazyData.select(pl.col(name) for name in colName)


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = SampleViewCreate_test(lazyData)
    result = pl.LazyFrame(results)
    results = MeasFaultSplit(result)
    ### saving into csv format
    #results.to_csv(savingCSVPath)
    #print('work done.')
    #results = results.select(pl.col(Nm) for Nm in results.columns if Nm != 'Fault')
    return results



##### FIRST PREPROCESSING - END #####





##### SECOND PREPROCESSING #####



#def SecondRestruct(dataset, cartIdSet, engineId, timeId, RUL_Id, oldNm, folderPth, FileName, cartIdFileName):
def SecondRestruct(dataset, cartIdSet, engineId, timeId, RUL_Id, oldNm):
    try:
        newNm = ['sensor measurement 1', engineId, timeId, 'sensor measurement 2', 'sensor measurement 3', 'sensor measurement 4', 'sensor measurement 5', RUL_Id]
        
        #path = os.path.join(folderPth, FileName)
        #data = pl.scan_csv(path, separator = ',', has_header = True, new_columns = newNm)
        colDict = dict()
        for a in range(len(oldNm)):
            colDict[oldNm[a]] = newNm[a]
        data = dataset.rename(colDict)
        
        unitValues = data.select(pl.col(engineId))
        unitValues = unitValues.collect()
        unitValues = unitValues.to_pandas()
        unitValues = unitValues.iloc[:,0].tolist()
        unitValues = set(unitValues)
        frames = list()
        for engineVal in unitValues:
            subData = data.filter(pl.col(engineId) == engineVal)
            subData = subData.collect()
            subData = subData.to_pandas()
            size = subData.shape[0]
            timeList = list()
            for t in range(1, (size + 1)):
                timeList.append(t)
            subData[timeId] = timeList
            subData = pl.LazyFrame(subData)
            frames.append(subData)
        newDf = pl.concat(frames, how = 'vertical')
        
        
        newDf = newDf.with_columns(pl.col(RUL_Id) / 1000000)
        RUL_col = newDf.select(pl.col(RUL_Id))
        engine_col = newDf.select(pl.col(engineId))
        time_col = newDf.select(pl.col(timeId))
        #cartId_col = newDf.select(pl.col(newNm[0]))
        oth_cols = newDf.select(pl.col(Nm) for Nm in newDf.collect_schema().names() if ((Nm != engineId) & (Nm != RUL_Id) & (Nm != timeId) ))
        
        cols = [newNm[0]]
        for i in range(3,7):
            cols.append(newNm[i])
        scaler = MinMaxScaler(feature_range = (0,1))
        scaled_data = scaler.fit_transform(oth_cols.collect())
        scaled_data = pd.DataFrame(scaled_data, columns = cols)
        oth_cols = pl.LazyFrame(scaled_data)
        #CartIdScaler = MinMaxScaler(feature_range = (0,1))
        #CartIdScaled = CartIdScaler.fit_transform(cartId_col.collect())
        #CartIdScaled = pd.DataFrame(CartIdScaled, columns = newNm[0])
        #cartId_col = pl.LazyFrame(CartIdScaled)
        newDf = pl.concat([RUL_col, engine_col, time_col, oth_cols], how = 'horizontal')
        cartNames = cartIdSet.collect().to_pandas().iloc[:,0].to_numpy().tolist()
        DATA = data.select(pl.col(newNm[0])).collect().to_pandas()
        NEW_DATA = newDf.select(pl.col(newNm[0])).collect().to_pandas()
        rangeStop = DATA.shape[0]
        newCartNames = list()
        for cartName in cartNames:
            flag = True
            index = 0
            while flag == True:
                if( (DATA.iloc[index,0] == cartName) | (str(DATA.iloc[index,0]) == str(cartName)) ):
                    newCartNames.append(NEW_DATA.iloc[index,0])
                    flag = False
                else:
                    index = index + 1
        newCartNames = pd.DataFrame(newCartNames)
        newCartNames = pl.LazyFrame(newCartNames)
        newCartNames = newCartNames.rename({"0": 'New Cart Id - Norm'})
        newCartId = pl.concat([newCartNames, cartIdSet], how = 'horizontal')
        
        #SvFileName = FileName.rsplit('.', 1)[0]
        #SvFileName = SvFileName + ' - restructured.csv'
        #svPth = os.path.join(folderPth, SvFileName)
        #newDf.collect().write_csv(svPth, separator = ',')
        
        #SvFileName = cartIdFileName.rsplit('.', 1)[0]
        #SvFileName = SvFileName + ' - restructured.csv'
        #svPth = os.path.join(folderPth, SvFileName)
        #newCartId.collect().write_csv(svPth, separator = ',')
        return newDf, newCartId, 0
    except Exception as e:
        error_message = traceback.format_exc()
        print("Errore catturato:")
        print(error_message)
        return None, None, -1



def RULmax(df, maxVal):
    rulColumn = df.collect().to_numpy().tolist()
    rul = list()
    for i in range(len(rulColumn)):
        val = rulColumn[i][0]
        rul.append(val)
    new_rul = list()
    for value in rul:
        if value > maxVal:
            new_rul.append(1)
        else:
            new_rul.append((value / maxVal))
    new_rul_df = pd.DataFrame({'RUL': new_rul})
    return new_rul_df



def DataRULRestruct(dataset, RUL_id, engineId, maxRUL, filterRUL):
    try:
        #data = pl.scan_csv(path, separator = ',', has_header = True)
        #data = dataset.filter(pl.col(RUL_id) > filterRUL)
        data = dataset.group_by(engineId).agg(pl.len().alias('RUL length'))
        unitVals = data.filter(pl.col('RUL length') > filterRUL).select(pl.col(engineId))
        unitVals = unitVals.collect()
        unitVals = unitVals.to_pandas()
        unitVals = unitVals.iloc[:,0].tolist()
        unitVals = set(unitVals)
        data = list()
        for val in unitVals:
            subData = dataset.filter(pl.col(engineId) == val)
            data.append(subData)
        data = pl.concat(data, how = 'vertical')
        newRUL = RULmax(data, maxRUL)
        data = data.collect()
        data = data.to_pandas()
        data[RUL_id] = newRUL
        data = pl.LazyFrame(data)
        unitValues = data.select(pl.col(engineId))
        unitValues = unitValues.collect()
        unitValues = unitValues.to_pandas()
        unitValues = unitValues.iloc[:,0].tolist()
        unitValues = set(unitValues)
        return len(unitValues), data, 0
        
    except Exception as e:
        error_message = traceback.format_exc()
        print("Errore catturato:")
        print(error_message)
        return None, -1



def pesca_e_rimuovi(valori):
    if not valori:
        raise ValueError("L'insieme dei valori e vuoto, non ci sono valori da pescare.")
    valore_estratto = random.choice(valori)
    valori.remove(valore_estratto)
    return valore_estratto, valori



def splitting(lf, test_size, eng_col, time_col, numEngines):
    try:
        numTestEngines = int(numEngines*test_size)
        unitValues = lf.select(pl.col(eng_col))
        unitValues = unitValues.collect()
        unitValues = unitValues.to_pandas()
        unitValues = unitValues.iloc[:,0].tolist()
        unitValues = list(set(unitValues))
        testData = list()
        for i in range(numTestEngines):
            engVal, unitValues = pesca_e_rimuovi(unitValues)
            subData = lf.filter(pl.col(eng_col) == engVal)
            testData.append(subData)
        test = pl.concat(testData, how = 'vertical')
        train = lf.join(test, on = [eng_col, time_col], how = 'anti')
        return train, test, 0
    except Exception as e:
        error_message = traceback.format_exc()
        print("Errore catturato:")
        print(error_message)
        return None, None, -1



def OLDsplitting(lf, test_size, target_col):
    try:
        # Converte il LazyFrame in un DataFrame di Pandas
        df = lf.collect().to_pandas()
        # Separare le caratteristiche dal target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        # Effettua lo splitting utilizzando train_test_split (for simplify I fix random_state)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        # Ricrea i DataFrame di Pandas con il target incluso
        train_df = pd.concat([y_train, X_train], axis=1)
        test_df = pd.concat([y_test, X_test], axis=1)
        # Converti i DataFrame di Pandas di nuovo in LazyFrame di Polars
        train_lf = pl.DataFrame(train_df).lazy()
        test_lf = pl.DataFrame(test_df).lazy()
        return train_lf, test_lf, 0
    except Exception as e:
        error_message = traceback.format_exc()
        print("Errore catturato:")
        print(error_message)
        return None, None, -1



##### SECOND PREPROCESSING - END #####






##### ENCODING PHASE #####



#encoding a col
def ordEncoding(col):
    colRestr = col.collect().to_pandas()
    oe = OrdinalEncoder()
    oe.fit(colRestr)
    encodedCol = oe.transform(colRestr)
    encodedCol = [numero for sublist in encodedCol for numero in sublist]
    return encodedCol


def dataEncoding(data, ColExcpt, colToSave):
    columns = data.collect_schema().names()
    idCol = dict()
    for colNm in columns:
        col = data.select(pl.col(colNm))
        if (col.collect_schema().dtypes()[0] == pl.String) & (colNm != ColExcpt):
            encCol = ordEncoding(col)
            if colNm == colToSave:
                pandasCol = col.collect().to_pandas()
                for name in range(len(encCol)):
                    idCol[pandasCol.iloc[name,0]] = str(encCol[name])
            colNamesWithoutEncCol = [nm for nm in columns if nm != colNm]
            data = data.select(colNamesWithoutEncCol).with_columns(**{colNm : pl.Series(colNm,encCol)})
    
    #sorting the cols
    data = data.select(col for col in columns)
    
    newData = data
    return newData, idCol



##### ENCODING PHASE - END #####






##### ADDING NUM ENGINE'S COL #####

def addNumEnCol(data):
    faultCol = data.select(pl.col('Fault')).collect().to_pandas()
    colLen = faultCol.shape[0]
    units = list()
    unit = 1
    for i in range(colLen - 1, -1, -1):
        if faultCol.iloc[i, 0] == 0:
            units.append(unit)
        else:
            units.append(unit)
            unit = unit + 1
    units.reverse()
    
    restrData = data.select(pl.all()).with_columns(**{'engine' : pl.Series('engine',units)})
    newOrd = ['cart Id', 'engine', 'time', 'descr', 'Meas', 'Fault']
    restrData = restrData.select(newOrd)
    restrData = restrData.reverse()
    return restrData



##### ADDING NUM ENGINE'S COL - END #####






##### TIME RESTRUCT #####

def diffDate(date1,date2):
    dayDiff = (date2 - date1).days
    secDiff = (date2 - date1).seconds
    secDays = float(dayDiff * 24 * 3600)
    totSec = secDays + secDiff
    delta = timedelta(seconds=totSec)
    return delta

def timeRestruct(data):
    data = data.reverse()
    timeCol = data.select(pl.col('time'))
    subData = data.select(pl.col('engine')) #([pl.col('cart Id'), pl.col('engine'), pl.col('time'), pl.col('Fault')])
    subDataGrby = subData.group_by('engine').count()    
    pdData = subDataGrby.collect().to_pandas()

    max_unit_num = pdData['engine'].max() 
    min_unit_num = pdData['engine'].min() 
    number_of_units = int( max_unit_num - min_unit_num + 1 )
    
    final = dict()

    for unit in range(number_of_units):
        windows_unit = list()
        unitData = data.filter(pl.col('engine') == float(unit + min_unit_num)).select(pl.col('time'), pl.col('Fault')).collect().to_pandas()
        unitLen = unitData.shape[0]
        zeroTime = unitData.iloc[0,0]
        for i in range(0,unitLen):
            oldTime = unitData.iloc[i,0]
            newTime = diffDate(oldTime,zeroTime)
            windows_unit.append(newTime)
        final.update( {float(unit+min_unit_num):windows_unit} )

    #creation of lazyFrame
    listOfKeyVal = []
    for key, values in final.items():
        newVals = []
        for value in values:
            newVals.append(value)
        newVals.reverse()
        for value in newVals:
            listOfKeyVal.append((key, value))
    lf = pl.DataFrame(listOfKeyVal, [('engine', pl.Int64), ('life', pl.Duration)]).lazy() 
    
    #adding time col
    timeCol = timeCol.collect().to_numpy()
    temp = list()
    timeColLen = timeCol.shape[0]
    for j in range((timeColLen - 1), -1, -1):
        temp.append(timeCol[j,0])
    timeCol = np.array(temp)
    timeCol = pl.Series(timeCol, dtype = pl.Datetime)
    timeCol = timeCol.dt.replace_time_zone("UTC")
    lf = lf.with_columns(**{'time' : timeCol})
    
    #join for the output
    joined = data.join(lf, on=['engine', 'time'])
    
    names = joined.collect_schema().names()
    joined = joined.select(pl.col(nm) for nm in names if nm != 'Fault')
    
    return joined
    

def groupByTimeIntervals(data, rangeInMinutes, dropNA):
    
    #cisei = True
    #ciseiquasi = True
    # Imposta il fuso orario UTC
    utc = pytz.utc 
    ## groupby
    timeCol = data.select(pl.col('time'))
    subData = data.select(pl.col('engine'))
    subDataGrby = subData.group_by('engine').len()    
    pdData = subDataGrby.collect().to_pandas()

    max_unit_num = pdData['engine'].max() 
    min_unit_num = pdData['engine'].min() 
    number_of_units = int( max_unit_num - min_unit_num + 1 )
    unitRows = list()    
    for unit in range(number_of_units):
        unitData = data.filter(pl.col('engine') == float(unit + min_unit_num)).collect().to_pandas()
        #print('UNIT DATA:')
        #print(unitData)
        unitLength = unitData.shape[0]
        rowIndex = 0
        cartId = unitData.iloc[rowIndex,0]
        firstProp = True
        while (rowIndex in range(unitLength)):           
            firstDate = datetime.fromtimestamp(unitData.iloc[rowIndex,2].replace(tzinfo=utc).timestamp(), tz=utc)
            #print('FIRST DATE:')
            #print(firstDate)
            delta = timedelta(minutes=rangeInMinutes)
            lastDate = firstDate + delta
            #creating raw
            newRow = []
            newRow.append(cartId)
            newRow.append((unit + 1))
            newRow.append(firstDate)
            
            #reading values of this row
            temperCol = np.empty((0,))
            humidCol = np.empty((0,))
            battCol = np.empty((0,))
            temperBattCol = np.empty((0,))
            secondProp = True
            while((rowIndex < unitLength) & (secondProp)):
                #print('sei nel while')
                #print('current index = ' + str(rowIndex))
                currentTime = datetime.fromtimestamp(unitData.iloc[rowIndex,2].replace(tzinfo=utc).timestamp(), tz=utc)
                if currentTime <= lastDate:
                    measNm = str(unitData.iloc[rowIndex,3])
                    value = float(unitData.iloc[rowIndex,4])
                    if measNm == 'TEMPERATURE':
                        temperCol = np.append(temperCol, value)
                    if measNm == 'HUMIDITY':
                        humidCol = np.append(humidCol, value)
                    if measNm == 'BATTERY_PERCENTAGE':
                        #if cisei:
                            #print('Hooray')
                            #cisei = False
                        battCol = np.append(battCol, value)
                    if measNm == 'BATTERY_TEMPERATURE':
                        temperBattCol = np.append(temperBattCol, value)
                    rowIndex = rowIndex + 1
                else:
                    #print('questo giro niente')
                    secondProp = False            
            #print('HUMIDITY VALUES:')
            #print(humidCol)
            #print('TEMPERATURE VALUES:')
            #print(temperCol)
            #print('INDEX:' + str(rowIndex))
            #if unit >= 4:
                #exit()
            # values calculated by mean
            # temperature
            if np.size(temperCol) != 0:
                temperMean = np.mean(temperCol, dtype=np.float64)
            else:
                temperMean = float('nan')
            # humidity
            if np.size(humidCol) != 0:
                humidMean = np.mean(humidCol, dtype=np.float64)
            else:
                humidMean = float('nan')
            # battery percentage of charge
            if np.size(battCol) != 0:
                #if ciseiquasi:
                    #print('HOORAY!')
                    #ciseiquasi = False
                battMean = np.mean(battCol, dtype=np.float64)
            else:
                battMean = float('nan')
            # battery temperature
            if np.size(temperBattCol) != 0:
                temperBattMean = np.mean(temperBattCol, dtype=np.float64)
            else:
                temperBattMean = float('nan')
        
            # insert values in row
            newRow.append(temperMean)
            newRow.append(humidMean)
            
            ### ONLY IF WE ACCESS TO THEESE INFO
            newRow.append(battMean)
            newRow.append(temperBattMean)
            
            # add new row in list of rows
            unitRows.append(newRow)
            #print('RIGA AGGIUNTA:')
            #print(unitRows)
            #print('Ora dovresti ritornare all\'inizio del for, a partir dall\'indice ' + str(rowIndex) + ' ricalcolo first e last date, e provo a rientrare nel while')
            #print('riga accorpata')
            #print(unitRows)
    # col names
    lzColumns = ['cart Id', 'engine', 'time', 'TEMPERATURE', 'HUMIDITY', 'BATTERY', 'BATTERY_TEMPERATURE']
    lzSchema = {lzColumns[0]: pl.Float64, lzColumns[1]: pl.Int64, lzColumns[2]: pl.Datetime, lzColumns[3]: pl.Float64, lzColumns[4]: pl.Float64, lzColumns[5]: pl.Float64, lzColumns[6]: pl.Float64}
    dfLazy = pl.LazyFrame({col_name: [row[i] for row in unitRows] for i, col_name in enumerate(lzColumns)}, schema = lzSchema)
    # Conversione della colonna 'time' in formato datetime
    #lazy_df = lazy_df.with_columns(pl.col('time').cast(pl.Datetime))
    #print(lazy_df.collect())    
    cond1 = pl.col('TEMPERATURE') == float('nan')
    cond2 = pl.col('HUMIDITY') == float('nan')
    cond3 = pl.col('BATTERY') == float('nan')
    cond4 = pl.col('BATTERY_TEMPERATURE') == float('nan')
    totCond = cond1 & cond2 & cond3 & cond4 
    dfLazy = dfLazy.with_columns([pl.when(totCond).then(1.0).otherwise(0.0).alias('Fault')])
    
    if dropNA:
        # create null values (cast from nan to null)
        dfLazy = dfLazy.fill_nan(None)
        # remove rows with at least one null value
        dfLazy = dfLazy.drop_nulls()
    return dfLazy
    ##### COMMENTO #####
    # Qui ho i dati in due liste, devo decidere come inserirli nel db
    # ho bisogno di inserire 4 colonne
    # una per il tempo, e tre per le misure
    # battery,Temper, Humid
    # decidere i nullVal come gestirli:
    #   - scrivere NULL
    #   - mettere un val di default
    # decidere i measVal come gestirli
    # decidere come e dove salvarli...
    # una volta inseriti, si ricomincia!



def groupByTimeIntervals_test(data, rangeInMinutes, dropNA):
    
    #cisei = True
    #ciseiquasi = True
    # Imposta il fuso orario UTC
    utc = pytz.utc 
    ## groupby
    timeCol = data.select(pl.col('time'))
    subData = data.select(pl.col('cart Id'))
    subDataGrby = subData.group_by('cart Id').len()    
    pdData = subDataGrby.collect().to_pandas()

    max_unit_num = pdData['cart Id'].max() 
    min_unit_num = pdData['cart Id'].min() 
    number_of_units = int( max_unit_num - min_unit_num + 1 )
    #print(number_of_units)
    #exit()
    unitRows = list()    
    for unit in range(number_of_units):
        unitData = data.filter(pl.col('cart Id') == float(unit + min_unit_num)).collect().to_pandas()
        #print('UNIT DATA:')
        #print(unitData)
        unitLength = unitData.shape[0]
        rowIndex = 0
        cartId = unitData.iloc[rowIndex,0]
        firstProp = True
        while (rowIndex in range(unitLength)):           
            firstDate = datetime.fromtimestamp(unitData.iloc[rowIndex,2].replace(tzinfo=utc).timestamp(), tz=utc)
            #print('FIRST DATE:')
            #print(firstDate)
            delta = timedelta(minutes=rangeInMinutes)
            lastDate = firstDate + delta
            #creating raw
            newRow = []
            newRow.append(cartId)
            newRow.append((unit + 1))
            newRow.append(firstDate)
            
            #reading values of this row
            temperCol = np.empty((0,))
            humidCol = np.empty((0,))
            battCol = np.empty((0,))
            temperBattCol = np.empty((0,))
            secondProp = True
            while((rowIndex < unitLength) & (secondProp)):
                #print('sei nel while')
                #print('current index = ' + str(rowIndex))
                currentTime = datetime.fromtimestamp(unitData.iloc[rowIndex,2].replace(tzinfo=utc).timestamp(), tz=utc)
                if currentTime <= lastDate:
                    measNm = str(unitData.iloc[rowIndex,3])
                    value = float(unitData.iloc[rowIndex,4])
                    if measNm == 'TEMPERATURE':
                        temperCol = np.append(temperCol, value)
                    if measNm == 'HUMIDITY':
                        humidCol = np.append(humidCol, value)
                    if measNm == 'BATTERY_PERCENTAGE':
                        #if cisei:
                            #print('Hooray')
                            #cisei = False
                        battCol = np.append(battCol, value)
                    if measNm == 'BATTERY_TEMPERATURE':
                        temperBattCol = np.append(temperBattCol, value)
                    rowIndex = rowIndex + 1
                else:
                    #print('questo giro niente')
                    secondProp = False            
            #print('HUMIDITY VALUES:')
            #print(humidCol)
            #print('TEMPERATURE VALUES:')
            #print(temperCol)
            #print('INDEX:' + str(rowIndex))
            #if unit >= 4:
                #exit()
            # values calculated by mean
            # temperature
            if np.size(temperCol) != 0:
                temperMean = np.mean(temperCol, dtype=np.float64)
            else:
                temperMean = float('nan')
            # humidity
            if np.size(humidCol) != 0:
                humidMean = np.mean(humidCol, dtype=np.float64)
            else:
                humidMean = float('nan')
            # battery percentage of charge
            if np.size(battCol) != 0:
                #if ciseiquasi:
                    #print('HOORAY!')
                    #ciseiquasi = False
                battMean = np.mean(battCol, dtype=np.float64)
            else:
                battMean = float('nan')
            # battery temperature
            if np.size(temperBattCol) != 0:
                temperBattMean = np.mean(temperBattCol, dtype=np.float64)
            else:
                temperBattMean = float('nan')
        
            # insert values in row
            newRow.append(temperMean)
            newRow.append(humidMean)
            
            ### ONLY IF WE ACCESS TO THEESE INFO
            newRow.append(battMean)
            newRow.append(temperBattMean)
            
            # add new row in list of rows
            unitRows.append(newRow)
            #print('RIGA AGGIUNTA:')
            #print(unitRows)
            #print('Ora dovresti ritornare all\'inizio del for, a partir dall\'indice ' + str(rowIndex) + ' ricalcolo first e last date, e provo a rientrare nel while')
            #print('riga accorpata')
            #print(unitRows)
    # col names
    lzColumns = ['cart Id', 'engine', 'time', 'TEMPERATURE', 'HUMIDITY', 'BATTERY', 'BATTERY_TEMPERATURE']
    lzSchema = {lzColumns[0]: pl.Float64, lzColumns[1]: pl.Int64, lzColumns[2]: pl.Datetime, lzColumns[3]: pl.Float64, lzColumns[4]: pl.Float64, lzColumns[5]: pl.Float64, lzColumns[6]: pl.Float64}
    dfLazy = pl.LazyFrame({col_name: [row[i] for row in unitRows] for i, col_name in enumerate(lzColumns)}, schema = lzSchema)
    # Conversione della colonna 'time' in formato datetime
    #lazy_df = lazy_df.with_columns(pl.col('time').cast(pl.Datetime))
    #print(lazy_df.collect())    
    cond1 = pl.col('TEMPERATURE') == float('nan')
    cond2 = pl.col('HUMIDITY') == float('nan')
    cond3 = pl.col('BATTERY') == float('nan')
    cond4 = pl.col('BATTERY_TEMPERATURE') == float('nan')
    totCond = cond1 & cond2 & cond3 & cond4 
    dfLazy = dfLazy.with_columns([pl.when(totCond).then(1.0).otherwise(0.0).alias('Fault')])
    
    if dropNA:
        # create null values (cast from nan to null)
        dfLazy = dfLazy.fill_nan(None)
        # remove rows with at least one null value
        dfLazy = dfLazy.drop_nulls()
    return dfLazy



def dropNAN_(data):
    # create null values (cast from nan to null)
    restrData = data.fill_nan(None)
    # remove rows with at least one null value
    restrData = restrData.drop_nulls()
    return restrData



##### TIME RESTRUCT - END #####



##### CART FILTERING #####
def cart_filter(cartData, serial, dataset, size):
    try:
        subData = cartData.filter(pl.col('Old Cart Id') == serial)
        subData = subData.collect()
        subData = subData.to_pandas()
        if len(subData.shape) == 2:
            newId = float(subData.iloc[0,0])
        elif len(subData.shape) == 1:
            newId = float(subData.iloc[0])
        #print(newId)
        #print(dataset.select(pl.col('sensor measurement 1')).collect())
        subData = dataset.filter(pl.col('sensor measurement 1') == newId)
        subData = subData.collect()
        subData = subData.to_pandas()
        subData = subData.iloc[-size:,:]
        subData = pl.LazyFrame(subData)
        return subData
    except Exception as e:
        error_message = traceback.format_exc()
        print("Errore catturato:")
        print(error_message)
        return None
    


########## PREPROCESSING - END ##########











########## CREATION WINDOWS ##########



def numCalculate(data, size):
    try:
        dataset = data.collect().to_pandas()
        num = dataset.shape[0] // size
        return num
    except Exception as e:
        print(e)
        traceback.print_exc()
        return -1



def filtering(data,size,numFilter):
    length = data.shape[0]
    cols = data.columns
    windows = list()
    for indRow in range(0,length,size):
        window = data.iloc[indRow:(indRow+size),:]
        newWindow = pd.DataFrame(columns = cols)
        i = 0
        for index in range(0,size,numFilter):
            newWindow.loc[i] = window.iloc[index,:]
            i = i + 1
        windows.append(newWindow)
    newData = pd.concat(windows, ignore_index = True)
    return newData
    


def windowsCreate(dataframe, index_name, time_name, length, step, descr, numFilter):

    '''
    fissiamo la finestra a 50, devo leggere le righe del motore uno da 0 a 49 e realizzare la finestra;
    dopodiche leggo le righe da 1 a 50; da 2 a 51; ... da len_mot1-50 a len_mot1;
    per il motore 1 ho creato le finestre. le creo per tutti i motori, ottenendo per ogni motore un numero 
    indefinito di finestre.
    '''
    

    performance_text = 'Fase 1 - Data Merging\n'
    performance_text += 'Hello, In this file we report results of restructuring ' + descr +' windows. Starting now!\n'
        
    timeStart = ti.process_time()
    _groupedby_unit = dataframe.group_by(index_name).agg(pl.len().alias('life'))
    _groupedby_unit = _groupedby_unit.select(pl.all()).collect().to_pandas()
    old_unitCol = _groupedby_unit.iloc[:,0].tolist()
    old_unitLength = _groupedby_unit.iloc[:,1].tolist()
    numberOfUnits = len(old_unitCol)
    #### removing units of length less than length parameter
    
    #print(old_unitCol)
    #print(old_unitLength)
    indexes = list()
    for index in range(numberOfUnits):
       if old_unitLength[index] <= 30:
            indexes.append(index)
    unitCol = list()
    unitLength = list()
    for i in range(numberOfUnits):
        prop = True
        for index in indexes:
            if i == index:
                prop = False
        if prop == True:
            unitCol.append(old_unitCol[i])
            unitLength.append(old_unitLength[i])
    #print(unitCol)
    #print(unitLength)
    #exit()
    numberOfUnits = len(unitCol)
    number = 0
    windows_unit = list()
    final = list()
    for unit in range(numberOfUnits):
        windows_unit = list()
        un_len = unitLength[unit]
        if un_len >= length:
            for i in range(0,un_len,step):
                subData = dataframe.filter(pl.col(index_name) == unitCol[unit]).collect().to_pandas()
                if (i+length) <= un_len:
                    number = number + 1
                    window = subData.iloc[i:(i+length),:]
                    window = filtering(window, length, numFilter)
                    window = pl.LazyFrame(window)
                    window = window.select(pl.col(index_name).cast(pl.Int64), pl.col(time_name).cast(pl.Int64))
                    #window = window.with_columns(pl.col(index_name).cast(pl.Int64), pl.col(time_name).cast(pl.Int64), pl.col('RUL').cast(pl.Float64))
                    windows_unit.append(pl.LazyFrame(window.collect()))
                    #print(pl.concat(windows_unit, how = 'vertical').collect())
        if len(windows_unit) != 0:
            final.append( pl.LazyFrame(pl.concat(windows_unit, how = 'vertical').collect()) )
    if len(final) != 0:
        result = pl.concat(final, how = 'vertical')
    else:
        schema = {
        str(index_name): pl.Int64,  # Puoi cambiare il tipo di dati in base alle tue esigenze
        str(time_name): pl.Int64         # Puoi cambiare il tipo di dati in base alle tue esigenze
        }
        # Crea un LazyFrame vuoto
        result = pl.LazyFrame(schema=schema)
    timeEnd = ti.process_time()
    timeP = timeEnd - timeStart
    hour = timeP // 3600
    minu = (timeP - (hour*3600) ) // 60
    sec = (timeP - (hour*3600) - (minu * 60))
    performance_text += 'We have restructured ' + descr + ' windows (size ' + str(length) + ') in '
    performance_text += str(hour) + " hours, "
    performance_text += str(minu) + " minutes, "
    performance_text += str(sec) + " seconds."
    performance_text += "\n"
    return number, result, performance_text



def AllWindCreate(dataframe, index_name, time_name, length, step, numFilter):
    number = -1
    windResults = []
    perfdict = None
    try:
        number = dict()
        for key in dataframe.keys():
            descr = str(key)
            number[key], temporary_results, perfdict = windowsCreate(dataframe[key], index_name, time_name, length, step, descr, numFilter)
            windResults.append(temporary_results)
    except:
        error_message = traceback.format_exc()
        print("Errore catturato:")
        print(error_message)
    return number, windResults[0], windResults[1], perfdict




def ONLYAllWindCreate(dataframe, index_name, time_name, length, step, numFilter, svPth):
    try:
        WindowPth = createFld(svPth, 'Windows View')
        for key in dataframe.keys():
            if key == 'train':
                descr = str(key)
                number, train = windowsCreate(dataframe[key], index_name, time_name, length, step, svPth, descr, numFilter)
                name = str(length) + ' - Train.csv'
                savePlLazy(train, WindowPth, name)
                del(train)
                del(name)
            if key == 'test':
                descr = str(key)
                number, test = windowsCreate(dataframe[key], index_name, time_name, length, step, svPth, descr, numFilter)
                name = str(length) + ' - Test.csv'
                savePlLazy(test, WindowPth, name)
                del(test)
                del(name)
        return 0
    except Exception as e:
        print(e)
        traceback.print_exc()
        return -1



########## CREATION WINDOWS - END ##########











########## DEFINING MODEL ##########



@tf.keras.utils.register_keras_serializable()
def custom_loss(y_true, y_pred):
    # Calcola gli errori
    errors = y_pred - y_true

    # Definisci i limiti delle fasce
    threshold_1 = 3600
    threshold_2 = 3600 * 3
    
    # Condizione per penalizzare errori assoluti > 1200
    penalization_condition_1 = (errors) > 0
    penalization_condition_2 = tf.abs(errors) > 600
    penalization_condition_3 = tf.abs(errors) > 1200
    
    # Calcola i pesi per le fasce
    weight_1 = tf.where(y_true <= threshold_1, 40.0, 1.0)  # Peso maggiore per la prima fascia
    weight_2 = tf.where((y_true > threshold_1) & (y_true <= threshold_2), 20.0, 1.0)  # Peso medio per la seconda fascia
    weight_3 = tf.where(y_true > threshold_2, 10.0, 1.0)  # Peso minore per la terza fascia

    # Penalizza errori in eccesso
    weighted_errors = tf.where(penalization_condition_1, errors * weight_1, errors)  # Penalizza errori in eccesso nella prima fascia
    weighted_errors = tf.where((penalization_condition_2) & (y_true > threshold_1) & (y_true <= threshold_2), errors * weight_2, weighted_errors)  # Penalizza errori in eccesso nella seconda fascia
    weighted_errors = tf.where((penalization_condition_3) & (y_true > threshold_2), errors * weight_3, weighted_errors)  # Penalizza errori in eccesso nella terza fascia

    # Calcola la perdita media con pesi maggiori per gli errori in eccesso nella fascia critica
    loss = tf.reduce_mean(tf.square(weighted_errors))
    
    return loss





def createModel(numLSTMLay, numGRULay, n_neurons, n_batch_size, dropRate, shape, learning_rate):
    

    # Definizione del learning rate
    #learning_rate = 0.001  # Puoi provare diversi valori, ad esempio 0.0001, 0.00001, ecc.

    # Creazione dell'ottimizzatore Adam con il learning rate specificato
    optimizer = Adam(learning_rate=learning_rate)
    model = Sequential()
    model.add(Input(shape = (shape[1],shape[2])))
    for i in range((numLSTMLay -1)):
        model.add(LSTM(units=n_neurons, return_sequences=True, kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(dropRate))

    for i in range(numGRULay):
        model.add(GRU(units=n_neurons, return_sequences=True, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(dropRate))
        
    model.add(LSTM(units=n_neurons, return_sequences=False, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(dropRate))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='relu'))
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mse']) #metrics=[tf.keras.metrics.RootMeanSquaredError()]
    
    return model



def CreatePossConfig(numLSTMLay, numGRULay, n_neurons, n_batch_size, dropRate):
    possConfig = list()
    for i in range(1, numLSTMLay):
        possibilities = [True, False]
        possConfig.append(possibilities)
    for i in range(0, numGRULay):
        possibilities = [True, False]
        possConfig.append(possibilities)
    possConfig.append([n_neurons])
    possConfig.append([n_batch_size])
    possConfig.append([dropRate])
    return possConfig



def configurations(config):
    possible_combinations = list(itertools.product(*config))
    return possible_combinations



def ModelCreating(numLSTMLay, numGRULay, n_neurons, n_batch_size, dropRate, numbWind, size, numFeat, learR):
    models = list()
    try:
        shape = [numbWind, size, numFeat]
        # index = 1
        for lay1 in range(numLSTMLay):
            for lay2 in range((numGRULay + 1)):
                our_model = createModel(lay1, lay2, n_neurons, n_batch_size, dropRate, shape, learR)
                models.append(our_model)
                '''
                ### creare cartella e inserire file dentro
                folder_name = str(index) + "th configuration"
                path = os.path.join(svPath, 'models')
                if not os.path.exists(path): 
                    os.mkdir(path)
                folder_path = os.path.join(path,folder_name)
                if not os.path.exists(folder_path): 
                    os.mkdir(folder_path)
                model_name = 'lstm_model.keras'
                model_path = os.path.join(folder_path, model_name)
                #our_model.summary()
                our_model.save(model_path)
                text_file_name = str(index) + "th configuration.txt"
                text_path = os.path.join(folder_path,text_file_name)
                with open(text_path, 'w') as f:
                    f.write('Model created.')
                    f.write('\n')
                    f.write('It refers to the configuration: ')
                    f.write('\n')
                    f.write('---------------')
                    f.write('\n')
                    for a in range(lay1):
                        f.write('| LSTM  | True |')
                        f.write('\n')
                        f.write('| DROP  | True |')
                        f.write('\n')
                    for a in range(lay2):
                        f.write('| GRU   | True |')
                        f.write('\n')
                        f.write('| DROP  | True |')
                        f.write('\n')
                    f.write('| LSTM  | True |')
                    f.write('\n')
                    f.write('| DROP  | True |')
                    f.write('\n')
                    f.write('| DENSE | True |')
                    f.write('\n')
                    f.write('---------------')
                    f.write('\n')
                    f.write('\n')
                    f.write('NUMBER OF UNITS FOR EACH LAYER: ' + str(n_neurons))
                    f.write("\n")
                    f.write('BATCH SIZE FOR EACH LAYER: ' + str(n_batch_size))
                    f.write("\n")
                    f.write('DROPOUT RATE: ' + str(dropRate))
                    f.write("\n")
                    f.write('OPTIMIZER OF COMPILER: Adam')
                    f.write("\n")
                    f.write('SHAPE: [' + str(numbWind) + ', ' + str(size) + ', ' + str(numFeat) + ']')
                    f.write("\n")
                index = index + 1
                '''
        return models
    except Exception as e:
        error_message = traceback.format_exc()
        print("Errore catturato:")
        print(error_message)
        return None



def modelSummary(model):
    try:
        return model.summary()
    except Exception as e:
        print("An error occurred:", e)
        return -1



def carica_modello(file_path, num_config):
    """
    Carica un modello Keras da un file .keras.

    Args:
        file_path (str): Percorso del file .keras che contiene il modello.

    Returns:
        modello (keras.Model): Modello Keras caricato.
    """
    try:
        folder = 'models'
        conf_fld = str(num_config) + 'th configuration'
        model_nm = 'lstm_model.keras'
        model_path = os.path.join(file_path, folder) 
        conf_path = os.path.join(model_path, conf_fld)
        path = os.path.join(conf_path, model_nm)
        modello = load_model(path)
        #print("Modello caricato con successo da:", path)
        return modello
    except Exception as e:
        print("Errore durante il caricamento del modello:", str(e))
        return None

def carica_modello_test(file_path):
    """
    Carica un modello Keras da un file .keras.

    Args:
        file_path (str): Percorso del file .keras che contiene il modello.

    Returns:
        modello (keras.Model): Modello Keras caricato.
    """
    try:
        modello = load_model(file_path)
        #print("Modello caricato con successo da:", path)
        return modello
    except Exception as e:
        print("Errore durante il caricamento del modello:", str(e))
        return None

########## DEFINING MODEL ##########











########## TRAINING MODEL ##########



def error_evaluation(errori):
    
    # 1. Istogramma
    plt.hist(errori, bins=30, density=True, alpha=0.6, color='g')
    mu, std = stats.norm.fit(errori)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title('Istogramma degli errori')
    plt.show()
    
    # 2. Q-Q Plot
    stats.probplot(errori, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.show()
    
    # 3. Shapiro-Wilk Test
    shapiro_test = stats.shapiro(errori)
    print(f'Shapiro-Wilk Test: Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}')
    
    # 4. Kolmogorov-Smirnov Test
    ks_test = stats.kstest(errori, 'norm', args=(mu, std))
    print(f'Kolmogorov-Smirnov Test: Statistic={ks_test.statistic}, p-value={ks_test.pvalue}')
    
    # 5. Anderson-Darling Test
    ad_test = stats.anderson(errori, dist='norm')
    print('Anderson-Darling Test:', ad_test)
    return 0





def Bootstrapping(errori):

    try:
        # Numero di iterazioni per il bootstrapping
        n_iterations = 1000
        n_size = len(errori)
        
        # Bootstrapping
        bootstrapped_means = []
        for _ in range(n_iterations):
            sample = np.random.choice(errori, size=n_size, replace=True)
            bootstrapped_means.append(np.mean(sample))
        
        # Calcola intervalli di confidenza
        ranges = dict()
        
        alpha = 0.10
        lower_bound = np.percentile(bootstrapped_means, alpha/2*100)
        upper_bound = np.percentile(bootstrapped_means, (1-alpha/2)*100)
        ranges[alpha] = [lower_bound, upper_bound]
        
        
        alpha = 0.05
        lower_bound = np.percentile(bootstrapped_means, alpha/2*100)
        upper_bound = np.percentile(bootstrapped_means, (1-alpha/2)*100)
        ranges[alpha] = [lower_bound, upper_bound]
        
        
        alpha = 0.01
        lower_bound = np.percentile(bootstrapped_means, alpha/2*100)
        upper_bound = np.percentile(bootstrapped_means, (1-alpha/2)*100)
        ranges[alpha] = [lower_bound, upper_bound]
        
        
        return 0, ranges
    except Exception as e:
        print("Errore durante il training del modello:", str(e))
        return -1, None





def Training(data, trainSet, testSet, model, size, numFeat, engineID, timeID, RUL_Id, RUL_val, nEpochs, nBatch):
    try:
        train = trainSet.join(data['train'], on = [engineID, timeID] , how = 'inner')
        test = testSet.join(data['test'], on = [engineID, timeID] , how = 'inner')
        
        ### Cast from lazyframe to numpy
        npdata = train.collect().to_numpy()
        npdata2 = test.collect().to_numpy()
        
        ### reshape 
        npdata = npdata.reshape(npdata.shape[0] // size, size, npdata.shape[1])
        npdata2 = npdata2.reshape(npdata2.shape[0] // size, size, npdata2.shape[1])
        
        #loading and saving info
        '''
        PNG2Nm = '- training.png'
        PNGNm = '- summary.png'
        config_fld = str(num) + 'th configuration'
        svFilename = 'results.txt'
        model_name = 'model fitted.keras'
        fileName2 = 'loss function ' + PNG2Nm
        config_path = createFld(fldPath, config_fld)
        path = os.path.join(config_path, model_name)
        svTxTPath = os.path.join(config_path, svFilename)
        PNG2Path = os.path.join(config_path, fileName2)
        
        with open(svTxTPath, 'w') as f:
            f.write('Starting training...')
            f.write('\n')
        '''
        start = t.process_time()
        
        # prepare input data
        ### split
        time_data = npdata2[:,:,1]
        time_data = time_data[:,-1]
        x_train = npdata[:,:,3:]
        y_train = npdata[:,:,2]
        x_test = npdata2[:,:,3:]
        #x_test = x_test[:,-1,:]
        y_test_old = npdata2[:,:,2]
        y_test = y_test_old[:,-1]
        
        # fitting
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=2) #start_from_epoch = 30 , baseline = difficile da usare
        #mc = ModelCheckpoint(path, monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        # Callback per ridurre il learning rate quando la validazione loss non migliora
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        history = model.fit(x_train, y_train, epochs=nEpochs, batch_size=nBatch, callbacks=[es,reduce_lr], verbose=0, validation_split = 0.2)
        #history = model.fit(x_train, y_train, epochs=nEpochs, batch_size=nBatch, callbacks=[es, mc, reduce_lr],
        # verbose=0, validation_split = 0.2)

        # Crea il grafico
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        '''
        epochs_range = range(1, 100)
        fig = plt.figure()
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epochs', fontsize=40)
        # Specificare i valori dell'asse x
        x = np.linspace(0, 100, 100)
        xticks = np.linspace(min(x), max(x), 20)  # 20 valori sull'asse x
        plt.xticks(xticks)
        plt.ylabel('Loss', fontsize=40)
        plt.title('Training and Validation Loss', fontsize=50)
        plt.legend(loc='upper right', fontsize=25)
        
        # Impostazione delle dimensioni della figura
        fig.set_figwidth(15)  # Imposta la larghezza della figura su 15 pollici
        fig.set_figheight(10) # Imposta l'altezza della figura su 10 pollici
        '''
        #plt.savefig(PNG2Path)
        #plt.close()
        #plt.show()
        
        #R.U.L. cart's graphs
        cartIds = npdata2[:,:,0]
        cartIds = cartIds[:,-1]
        cartIds = list(set(cartIds))
        for cartId in cartIds:
            #fileName = 'Cart #' + str(cartId) + ' ' + PNGNm
            #PNGPath = os.path.join(config_path, fileName)
            indexes = np.where(npdata2[:,-1,0] == cartId)[0]
            cart_x_test = npdata2[indexes, :, 3:]
            cart_y_test = npdata2[indexes, :, 2]
            cart_y_test = cart_y_test[:, -1]
            cart_time_data = npdata2[indexes,:,1]
            cart_time_data = cart_time_data[:,-1]
            # Take model's predictions on validation data
            cart_y_pred = model.predict(cart_x_test)
            cart_y_pred = cart_y_pred[:,-1]
            
            # Recasting results respect prepr.-normalization
            cart_y_pred = cart_y_pred * RUL_val
            cart_y_test = cart_y_test * RUL_val
            rmse = list()
            # RMSE calculating
            if ( (cart_y_test.shape[0] != 0) & (cart_y_pred.shape[0] != 0) ):
                rmse.append( np.sqrt(np.mean((cart_y_test - cart_y_pred)**2)) )
            else:
                rmse.append(None)
            # Find index of R.U.L. values between range ( (3600*3) , 3600 )
            indices = np.where((cart_y_test <= (3600 * 3)) & (cart_y_test >= 3600))[0]
            # Take R.U.L. values
            trueVals = cart_y_test[indices]
            predVals = cart_y_pred[indices]
            if ( (trueVals.shape[0] != 0) & (predVals.shape[0] != 0) ):
                rmse.append( np.sqrt(np.mean((trueVals - predVals)**2)) )
            else:
                rmse.append(None)
            # Find index of R.U.L. values under 3600
            indices = np.where(cart_y_test <= 3600)[0]
            # Take R.U.L. values
            trueVals = cart_y_test[indices]
            predVals = cart_y_pred[indices]
            if ( (trueVals.shape[0] != 0) & (predVals.shape[0] != 0) ):
                rmse.append( np.sqrt(np.mean((trueVals - predVals)**2)) )
            else:
                rmse.append(None)

            '''
            #creation graph
            fig, ax = plt.subplots()
            # Choice color's graph        
            custom_colors = ['cyan', 'red', 'orange', 'purple', 'yellow', 'blue', 'magenta', 'lime', 'pink', 'teal', 'lavender', 'brown', 'beige', 'maroon', 'mint', 'olive', 'coral', 'navy', 'grey']
            test_color = custom_colors[0]
            pred_color = custom_colors[1]
            ax.scatter(cart_time_data, cart_y_pred, color=pred_color, label='Predicted #{} R.U.L.'.format(cartId))
            ax.scatter(cart_time_data, cart_y_test, color=test_color, label='True R.U.L.')
            ax.set_title('Engine #' + str(cartId), fontsize=50, fontweight='bold')
            ax.set_xlabel('Time', fontsize=40)
            ax.set_ylabel('R.U.L.', fontsize=40)
            #rmseValue = np.sqrt(np.mean((cart_y_test - cart_y_pred)**2))
            if rmse[0] is not None:
                ax.text(0.05, 0.30, 'Global RMSE: {:.2f}'.format(rmse[0]),
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes,
                color='black', fontsize=25, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
            if rmse[1] is not None:
                ax.text(0.05, 0.60, 'Middle RMSE: {:.2f}'.format(rmse[1]),
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes,
                color='black', fontsize=25, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
            if rmse[2] is not None:
                ax.text(0.05, 0.90, 'Last RMSE: {:.2f}'.format(rmse[2]),
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes,
                color='black', fontsize=25, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
            ax.legend(loc='upper right', fontsize=25)
            fig.set_size_inches(15, 10)
            plt.savefig(PNGPath)
            plt.close()
            del(fig)
            del(ax)
            '''
        
        y_pred = model.predict(x_test)
        y_pred = y_pred[:,-1]    
        rmse_y_test = y_test * RUL_val
        rmse_y_pred = y_pred * RUL_val
        errori = rmse_y_test - rmse_y_pred
        #error_evaluation(errori) not normal distribution
        confidenceRanges = list()
        flag, ranges = Bootstrapping(errori)
        confidenceRanges.append(ranges)
        
        
        # RMSE calculating
        rmse = list()
        if ( (rmse_y_test.shape[0] != 0) & (rmse_y_pred.shape[0] != 0) ):
            rmseVal = np.sqrt(np.mean((rmse_y_test - rmse_y_pred)**2))
        else:
            rmseVal = None
        rmse.append(rmseVal)
        
        # Find index of R.U.L. values between range ( (3600*3) , 3600 )
        indices = np.where((rmse_y_test <= (3600 * 3)) & (rmse_y_test >= 3600))[0]
        # Take R.U.L. values
        trueVals = rmse_y_test[indices]
        predVals = rmse_y_pred[indices]
        
        errori = trueVals - predVals
        #error_evaluation(errori)
        flag, ranges = Bootstrapping(errori)
        confidenceRanges.append(ranges)
        
        
        if ( (trueVals.shape[0] != 0) & (predVals.shape[0] != 0) ):
            rmseVal = np.sqrt(np.mean((trueVals - predVals)**2))
        else:
            rmseVal = None
        rmse.append(rmseVal)
        
        # Find index of R.U.L. values under 3600
        indices = np.where(rmse_y_test <= 3600)[0]
        # Take R.U.L. values
        trueVals = rmse_y_test[indices]
        predVals = rmse_y_pred[indices]
        
        errori = trueVals - predVals
        #error_evaluation(errori)
        flag, ranges = Bootstrapping(errori)
        confidenceRanges.append(ranges)
        
        if ( (trueVals.shape[0] != 0) & (predVals.shape[0] != 0) ):
            rmseVal = np.sqrt(np.mean((trueVals - predVals)**2))
        else:
            rmseVal = None
        rmse.append(rmseVal)
        
        
        # accuracy calculating
        accuracy = list()
        gl_accuracy = model.evaluate(x_test, y_test_old)
        accuracy.append(gl_accuracy[0])
        indices = np.where((rmse_y_test <= (3600 * 3)) & (rmse_y_test >= 3600))[0]
        # Find index of R.U.L. values between range ( (3600*3) , 3600 )
        trueVals = y_test_old[indices,:]
        xVals = x_test[indices, :, :]
        if ( (xVals.shape[0] != 0) & (trueVals.shape[0]) ):
            middle_accuracy = model.evaluate(xVals, trueVals)
        else:
            middle_accuracy = [None]
        accuracy.append(middle_accuracy[0])
        # Find index of R.U.L. values under 3600
        indices = np.where(rmse_y_test <= 3600)[0]
        # Take R.U.L. values
        trueVals = y_test_old[indices,:]
        xVals = x_test[indices, :, :]
        if ( (xVals.shape[0] != 0) & (trueVals.shape[0]) ):
            last_accuracy = model.evaluate(xVals, trueVals)
        else:
            last_accuracy = [None]
        accuracy.append(last_accuracy[0])
        end = t.process_time()
        time_elapsed = end - start
        time_elapsed = ti.strftime("%H:%M:%S", ti.gmtime(time_elapsed))

        '''    
        with open(svTxTPath, 'a') as f:
            f.write('...the test is completed (time elapsed: ' + str(time_elapsed) + ').')
            f.write('\n')
            f.write('Global R.M.S.E. : ' + str(rmse[0]) + ';\n')
            f.write('Global accuracy : ' + str(accuracy[0]) + '.\n')
            f.write('Confidence ranges : ')
            f.write('\n')
            lower_bound = confidenceRanges[0][0.10][0]
            upper_bound = confidenceRanges[0][0.10][1]
            f.write(f'90% of confidence : [{lower_bound}, {upper_bound}]')
            f.write('\n')
            lower_bound = confidenceRanges[0][0.05][0]
            upper_bound = confidenceRanges[0][0.05][1]
            f.write(f'95% of confidence : [{lower_bound}, {upper_bound}]')
            f.write('\n')
            lower_bound = confidenceRanges[0][0.01][0]
            upper_bound = confidenceRanges[0][0.01][1]
            f.write(f'99% of confidence : [{lower_bound}, {upper_bound}]')
            f.write('\n')
            f.write('\n')
            f.write('(from 3h to 1h of) R.U.L. R.M.S.E. : ' + str(rmse[1]) + ';\n')
            f.write('(from 3h to 1h of) R.U.L. accuracy : ' + str(accuracy[1]) + '.\n')
            f.write('Confidence ranges : ')
            f.write('\n')
            lower_bound = confidenceRanges[1][0.10][0]
            upper_bound = confidenceRanges[1][0.10][1]
            f.write(f'90% of confidence : [{lower_bound}, {upper_bound}]')
            f.write('\n')
            lower_bound = confidenceRanges[1][0.05][0]
            upper_bound = confidenceRanges[1][0.05][1]
            f.write(f'95% of confidence : [{lower_bound}, {upper_bound}]')
            f.write('\n')
            lower_bound = confidenceRanges[1][0.01][0]
            upper_bound = confidenceRanges[1][0.01][1]
            f.write(f'99% of confidence : [{lower_bound}, {upper_bound}]')
            f.write('\n')
            f.write('\n')
            f.write('(last 1h of) R.U.L. R.M.S.E. : ' + str(rmse[2]) + ';\n')
            f.write('(last 1h of) R.U.L. accuracy : ' + str(accuracy[2]) + '.\n')
            f.write('Confidence ranges : ')
            f.write('\n')
            lower_bound = confidenceRanges[2][0.10][0]
            upper_bound = confidenceRanges[2][0.10][1]
            f.write(f'90% of confidence : [{lower_bound}, {upper_bound}]')
            f.write('\n')
            lower_bound = confidenceRanges[2][0.05][0]
            upper_bound = confidenceRanges[2][0.05][1]
            f.write(f'95% of confidence : [{lower_bound}, {upper_bound}]')
            f.write('\n')
            lower_bound = confidenceRanges[2][0.01][0]
            upper_bound = confidenceRanges[2][0.01][1]
            f.write(f'99% of confidence : [{lower_bound}, {upper_bound}]')
            f.write('\n')
            f.write('\n')
        '''
        return 0, model, rmse[2], confidenceRanges
    except Exception as e:
        print("Errore durante il training del modello:", str(e))
        return -1, None, None

        
########## TRAINING MODEL - END ##########











########## TUNING MODEL ##########
def comparison(modelA, modelB):
    retval = modelA
    if (modelA[1] > modelB[1]):
        retval = modelB
    return retval

def Tuning(results, windResults_train, windResults_test, modelSize, numFeat, models, engineID, timeID, RUL_Id,
           maxRUL, Epochs, BatchSize):
    trained = list()
    try:
        index = 1
        for m in models:
            #if model is not None:
                # Eseguire ulteriori operazioni con il modello, ad esempio predizioni, valutazioni, ecc.
                #model.summary()
                
            print('ADDESTRAMENTO MODELLO DI PREVISIONE ' + str(index) + '/' + str(len(models)))
            index += 1
            #### TRAINING MODEL ####
            flag, trainedmodel, rmse, ranges = Training(results, windResults_train, windResults_test, m, modelSize,
                                                        numFeat, engineID, timeID, RUL_Id, maxRUL, Epochs, BatchSize)
            if flag == 0:
                print('CONCLUSO (rmse: ' + str(rmse) + ')')
            else:
                print('ERRORE NELLA FASE')
            info = (trainedmodel, rmse, ranges)
            trained.append(info)
        bestone = trained[0]
        for candidate in trained:
            bestone = comparison(bestone, candidate)
        return 0, bestone
    except Exception as e:
        print("Errore durante il tuning del modello:", str(e))
        return -1, None



def CreateModelAndTuning(data, train, test, sizeSamples, sizeWindows, sizeFeatures, cartName, timeName, targetName, numLSTMLay, numGRULay, numEpoc, units, batchSize, dropRate, learningRate, maxRUL, numConfigs, modelPath, trainingPath):
    try:
        txtName = 'tuning.txt'
        tuningPath = os.path.join(modelPath, txtName)
        with open(tuningPath, 'w') as f:
            f.write('Hello! Start our hyp. tuning now!')
            f.write('\n')
        
        # Devo creare dei range di varianza 
        # a partire dalle variabili di input:
        # numLSTMLay, numGRULay, numEpoc, units,
        # batchSize, dropRate, learningRate.
        
        ##### ESEMPIO - START #####
        # Definizione dei range delle variabili in base agli input forniti
        numLSTMLay_range = list(range(1, numLSTMLay + 1))
        numGRULay_range = list(range(1, numGRULay + 1))
        numEpoc_range = list(range(1, numEpoc + 1))
        units_range = [units // 2, units, units * 2]  # Varia intorno al valore dato
        batchSize_range = [batchSize // 2, batchSize, batchSize * 2]  # Varia intorno al valore dato
        dropRate_range = [dropRate - 0.1, dropRate, dropRate + 0.1]  # Varia intorno al valore dato
        learningRate_range = [learningRate / 10, learningRate, learningRate * 10]  # Varia intorno al valore dato
        
        # Numero di configurazioni da testare
        num_configs = 10
        
        # Ciclo per generare e testare le configurazioni casuali
        for i in range(num_configs):
            # Generazione casuale di una configurazione
            config = (random.choice(numLSTMLay_range),
                random.choice(numGRULay_range),
                random.choice(numEpoc_range),
                random.choice(units_range),
                random.choice(batchSize_range),
                random.choice(dropRate_range),
                random.choice(learningRate_range))
        ##### ESEMPIO - END #####

        
        
        
        # Dunque avr� una variabile configs, che conterr� 
        # diverse configurazioni delle variabili monitorate.
        
        # Dopodich� in un ciclo for (al variar della config),
        # lancio le funzioni: ModelCreating, Training
        # registro i risultati in termini di rmse (dunque modifica Training)
        # salvo i risultati nel file tuning.txt
        # (es. Configurazione #:   numEp = 123, batchSize = ...)
        return 0
    except Exception as e:
        print("Errore durante il tuning del modello:", str(e))
        return -1

