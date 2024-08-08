import time as ti
from src.core.functions import *
from src.utils.notifications import send_notification, llog


def model_train(trolleyids, Start, Stop, configuration, auxiliary_csv_filename):

    #### UPLOADING RANGES START AND STOP ####
    # Percorsi dei file di configurazione
    #*# training_config_path = '../../bucket/Training.ini'
    #*# testing_config_path = '../../bucket/Testing.ini'
    #*# main_config_path = '../../bucket/config.ini'
    #*# update_training_ini(training_config_path, Start, Stop)
    # Chiama la funzione per aggiornare le date
    #*# update_config_dates(training_config_path, main_config_path)
    #### STARTING TRAINING ####

    startPrepr = ti.process_time()
    llog('LETTURA PARAMETRI:')
    #### LOADING OF CONFIGURATION ####   
    url = configuration.get('influx_url')
    token = configuration.get('influx_tok')
    org = configuration.get('influx_org')
    bucket = configuration.get('influx_bucket')
    rangeInMin = configuration.get('rangeinmin')
    dropNAN = configuration.get('dropnan')
    nameFeat = configuration.get('namefeat')
    numFeat = configuration.get('numFeat')
    numLSTMLay = configuration.get('numlstmlay')
    numGRULay = configuration.get('numGRULay')
    batchSize = configuration.get('batchSize')
    units = configuration.get('units')
    learning_rate = configuration.get('learning_rate')
    dropRate = configuration.get('dropRate')
    numEpoc = configuration.get('numEpoc')
    engineId = configuration.get('newEngineNm')
    timeId = configuration.get('newTimeNm')
    RUL_Id =configuration.get('newRULNm')
    oldNames = [nameFeat[0],'engine','time', nameFeat[1], nameFeat[2], nameFeat[3], nameFeat[4], nameFeat[5]]
    maxRUL = configuration.get('maxRUL')
    filterRUL = configuration.get('threshRUL')
    percentOfSplit = configuration.get('percentOfSplit') / 100
    savedWindow = configuration.get('savedWindow')
    size = configuration.get('size')
    step = configuration.get('step')
    numFilter = configuration.get('numFilter')
    modelSize = size // numFilter
    possConfig = CreatePossConfig(numLSTMLay, numGRULay, units, batchSize, dropRate)
    numConfigs = len(configurations(possConfig))
    llog('CONCLUSA')
    send_notification('Inizio del test riferito alla folder')
    llog('CREAZIONE CARTELLA DI SALVATAGGIO:')
    #path = createFld(MthPth,FldNm)
    llog('CONCLUSA')

    using_influx_flag = auxiliary_csv_filename == None
    
    #### READING FROM KIRANET DB ####
    llog('LETTURA DATI DA SERVER:')
    if using_influx_flag:
        df = readingWithInfluxDB(bucket = bucket, org = org, token = token, url = url, start = Start, stop = Stop)
    else:
        df = readingCSVDB(auxiliary_csv_filename)
    llog('CONCLUSA')

    
    
    #### FIRST PREPROCESSING ####
    llog('PREPROCESSING:')
    results = Preprocessing(df)
    #SAVE
    #savePolarsLazyFrameCSV(results, SvNm, path)
    llog('CONCLUSO')

    results = results.filter(pl.col("cart Id").is_in(trolleyids))

    #### ENCODING CATEGORICAL VARIABLES AND FIX TIME ISSUES ####
    llog('GESTIONE VARIABILI CATEGORICHE:')
    results, cartId = dataEncoding(results, 'descr', 'cart Id')
    new_dict = {'New Cart Id': [], 'Old Cart Id': []}
    for key, value in cartId.items():
        new_dict['Old Cart Id'].append(key)
        new_dict['New Cart Id'].append(value)
    cartId = pl.LazyFrame(new_dict)#, columns=['New Cart Id', 'Old Cart Id'])
    #savePolarsLazyFrameCSV(cartId, cartIdNm, path)
    llog('CONCLUSA')
    
    llog('AGGIUNTA COLONNA ID CARRELLO:')
    results = addNumEnCol(results)
    llog('CONCLUSA')

    llog('RISTRUTTURAZIONE INFORMAZIONI TEMPORALI:')
    results = groupByTimeIntervals(results, rangeInMin, dropNAN)
    results = timeRestruct(results)
    results = dropNAN_(results)
    results = results.with_columns(pl.col('life').cast(pl.Int64))
    #savePolarsLazyFrameCSV(results, SvNm, path)
    llog('CONCLUSA')

    llog('SECONDA FASE DI PREPROCESSING:')
    dataset, cartId, flag = SecondRestruct(results, cartId, engineId, timeId, RUL_Id, oldNames)
    #old dataset, cartId, flag = SecondRestruct(results, cartId, engineId, timeId, RUL_Id, oldNames, path, SvNm,
    # cartIdNm)
    if flag == 0:
        llog('CONCLUSA')
    else:
        llog('ERRORE NELLA FASE')

    llog('TERZA FASE DI PREPROCESSING:')
    numCarts, dataset, flag = DataRULRestruct(dataset, RUL_Id, engineId, maxRUL, filterRUL)
    if flag == 0:
        llog('CONCLUSA')
    else:
        llog('ERRORE NELLA FASE')

    llog('QUARTA FASE DI PREPROCESSING: SPLITTING')
    results = dict()
    results['train'], results['test'], flag = splitting(dataset, percentOfSplit, engineId, timeId, numCarts)
    if flag == 0:
        llog('CONCLUSA')
    else:
        llog('ERRORE NELLA FASE')




    ##### FASE PER TESTARE IL CODICE - START #####
    SIMPLIFY = 1
    if SIMPLIFY == 1:
        #carico gli id dei carrelli
        trainValues = results['train'].select(pl.col(engineId))
        trainValues = trainValues.collect()
        trainValues = trainValues.to_pandas()
        trainValues = trainValues.iloc[:,0].tolist()
        trainValues = list(set(trainValues))
        testValues = results['test'].select(pl.col(engineId))
        testValues = testValues.collect()
        testValues = testValues.to_pandas()
        testValues = testValues.iloc[:,0].tolist()
        testValues = list(set(testValues))
        listOfCarts = dict()
        middList = list()
        # ne prendo 6 per la fase di training
        for i in range(3):
            value, trainValues = pesca_e_rimuovi(trainValues)
            middList.append(value)
        listOfCarts['train'] = middList
        # ne prendo 2 per la fase di testing
        middList = list()
        for i in range(2):
            value, testValues = pesca_e_rimuovi(testValues)
            middList.append(value)
        listOfCarts['test'] = middList
        # salvo tutto nella variabile apposita (un dict)
        for key in results.keys():
            data = results[key]
            results[key] = data.filter(pl.col(engineId).is_in(listOfCarts[key]))
    ##### FASE PER TESTARE IL CODICE - END #####

    


    llog('QUINTA FASE DI PREPROCESSING: CREAZIONE WINDOWS SETS')
    send_notification('QUINTA FASE DI PREPROCESSING: CREAZIONE WINDOWS SETS')
    temp = AllWindCreate(results, engineId, timeId, size, step, numFilter)
    number, windResults_train, windResults_test, processing_times = temp
    if type(number) != int:
        llog('CONCLUSA')
        send_notification('CONCLUSA')
    else:
        llog('ERRORE NELLA FASE')
        send_notification('ERRORE NELLA FASE')




    #### CREATION MODEL ####
    llog('CREAZIONE MODELLO DI PREVISIONE')
    
    model = ModelCreating(numLSTMLay, numGRULay, units, batchSize, dropRate, number, modelSize, numFeat, learning_rate)
    if model != 0:
        llog('CONCLUSA')
    else:
        llog('ERRORE NELLA FASE')

    llog('TUNING MODELLO DI PREVISIONE')
    send_notification('TUNING MODELLO DI PREVISIONE')
    flag, best = Tuning(results, windResults_train, windResults_test, modelSize, numFeat, model, engineId, timeId,
                        RUL_Id, maxRUL, numEpoc, batchSize)
    if flag == 0:
        llog('CONCLUSO')
        send_notification('CONCLUSO')
    else:
        llog('ERRORE NELLA FASE')
        send_notification('ERRORE NELLA FASE')
    return best, cartId   #### best[0] è una stringa, contiene il percorso per raggiungere il modello addestrato e poterlo richiamare