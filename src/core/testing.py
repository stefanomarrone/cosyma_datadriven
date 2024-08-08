########## IMPORTING NECESSARY LIBRARIES ##########
from src.core.functions import *
from src.utils.notifications import send_notification, llog


########## IMPORTING NECESSARY LIBRARIES - END ##########

def testing(configuration,model,ranges,cartId,serial,Start,Stop,auxiliary_csv_filename = None):
    print('LETTURA CONFIGURAZIONE:')
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
        

        if y_pred >= (3600*3):
            dev = ranges[0]
        if ( (y_pred <= (3600*3)) & (y_pred >= (3600)) ):
            dev = ranges[1]
        if y_pred <= (3600):
            dev = ranges[2]
        
        
        
        return y_pred, dev
    except Exception as e:
        error_message = traceback.format_exc()
        print("Errore catturato:")
        print(error_message)
        return None