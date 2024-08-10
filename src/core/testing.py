########## IMPORTING NECESSARY LIBRARIES ##########
from src.core.functions import *
from src.utils.notifications import send_notification, llog


########## IMPORTING NECESSARY LIBRARIES - END ##########

def testing(configuration, model, cartId, serial, Start, Stop, auxiliary_csv_filename=None):
    print('LETTURA CONFIGURAZIONE:')
    url = configuration.get('influx_url')
    token = configuration.get('influx_tok')
    org = configuration.get('influx_org')
    bucket = configuration.get('influx_bucket')
    rangeInMin = configuration.get('rangeinmin')
    dropNAN = configuration.get('dropnan')
    nameFeat = configuration.get('namefeat')
    numLSTMLay = configuration.get('numlstmlay')
    numGRULay = configuration.get('numGRULay')
    batchSize = configuration.get('batchSize')
    units = configuration.get('units')
    dropRate = configuration.get('dropRate')
    engineId = configuration.get('newEngineNm')
    timeId = configuration.get('newTimeNm')
    RUL_Id =configuration.get('newRULNm')
    oldNames = [nameFeat[0],'engine','time', nameFeat[1], nameFeat[2], nameFeat[3], nameFeat[4], nameFeat[5]]
    maxRUL = configuration.get('maxRUL')
    filterRUL = configuration.get('threshRUL')
    size = configuration.get('size')
    numFilter = configuration.get('numFilter')
    step = configuration.get('step')
    possConfig = CreatePossConfig(numLSTMLay, numGRULay, units, batchSize, dropRate)
    numFeat = configuration.get('numFeat')
    learning_rate = configuration.get('learning_rate')
    numEpoc = configuration.get('numEpoc')
    percentOfSplit = configuration.get('percentOfSplit') / 100
    savedWindow = configuration.get('savedWindow')
    modelSize = size // numFilter
    numConfigs = len(configurations(possConfig))

    using_influx_flag = auxiliary_csv_filename == None
    llog('LETTURA DATI DA SERVER:')
    if using_influx_flag:
        df = readingWithInfluxDB(bucket=bucket, org=org, token=token, url=url, start=Start, stop=Stop)
    else:
        df = readingCSVDB(auxiliary_csv_filename)
    llog('CONCLUSA')

    print('PREPROCESSING:')
    #### FIRST PREPROCESSING ####
    results = Preprocessing_test(df)
    print('CONCLUSO')

    serial = [serial]
    results = results.filter(pl.col("cart Id").is_in(serial))
    print('GESTIONE VARIABILI CATEGORICHE:')
    #### ENCODING CATEGORICAL VARIABLES AND FIX TIME ISSUES ####
    results, _ = dataEncoding(results, 'descr', 'cart Id')
    print('CONCLUSA')

    print('AGGIUNTA COLONNA ID CARRELLO:')
    results = addNumEnCol(results)
    print('CONCLUSA')
    
    print('RISTRUTTURAZIONE INFORMAZIONI TEMPORALI:')
    results = groupByTimeIntervals_test(results, rangeInMin, dropNAN)
    results = timeRestruct(results)
    results = dropNAN_(results)
    
    ### TO UPDATE ###
    results = results.with_columns(pl.col('life').cast(pl.Int64))
    ### TO UPDATE - END ###
    print('CONCLUSA')

    
    print('SECONDA FASE DI PREPROCESSING:')

    dataset, _, flag = SecondRestruct(results, cartId, engineId, timeId, RUL_Id, oldNames)
    if flag == 0:
        print('CONCLUSA')
    else:
        print('ERRORE NELLA FASE')

    print('TERZA FASE DI PREPROCESSING:')
    numCarts, dataset, flag = DataRULRestruct(dataset, RUL_Id, engineId, maxRUL, filterRUL)
    if flag == 0:
        print('CONCLUSA')
    else:
        print('ERRORE NELLA FASE')

    #### Filtering respect CART - ID SELECTED
    dataset = cart_filter(cartId, serial, dataset, size)

    try:
        test = dataset
        ### Cast from lazyframe to numpy
        npdata2 = test.collect().to_numpy()
        ### reshape 
        npdata2 = npdata2.reshape(npdata2.shape[0] // size, size, npdata2.shape[1])
        start = t.process_time()
        
        # prepare input data
        ### split
        time_data = npdata2[:,:,1]
        time_data = time_data[:,-1]
        x_test = npdata2[:,:,3:]
        y_pred = model.predict(x_test)
        y_pred = y_pred[:,-1]    
        y_pred = y_pred[0] * maxRUL
        return y_pred
    except Exception as e:
        error_message = traceback.format_exc()
        print("Errore catturato:")
        print(error_message)
        return None