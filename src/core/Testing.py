########## IMPORTING NECESSARY LIBRARIES ##########
from functions import *
########## IMPORTING NECESSARY LIBRARIES - END ##########

def testing(Start, Stop, serial):
    
    #### UPLOADING RANGES START AND STOP ####
    # Percorsi dei file di configurazione
    testing_config_path = '../../bucket/Testing.ini'
    main_config_path = '../../bucket/config.ini'
    # Chiama la funzione per aggiornare le date e il seriale
    update_testing_dates(testing_config_path, Start, Stop, serial)
    # Chiama la funzione per aggiornare le date
    update_config_dates_test(testing_config_path, main_config_path, serial)
    path = upload_path(testing_config_path)
    # Upload model
    model = carica_modello_test(path)
    
    
    #### STARTING TRAINING ####
    startPrepr = ti.process_time()
    print('LETTURA PARAMETRI:')
    #### LOADING OF CONFIGURATION ####   
    prova = Configuration('config.ini')
    url = prova.getConn('url')
    token = prova.getConn('token')
    org = prova.getConn('org')
    bucket = prova.getConn('bucket')
    serialNum = prova.getConn('serialNumber')
    RdNm = prova.getPar('RdNm')
    SvNm = prova.getPar('SvNm')
    cartIdNm = prova.getPar('CartSvNm')
    MthPth = prova.getPar('MthPth')
    FldNm = prova.getPar('FldNm')
    rangeInMin = prova.getMod('rangeInMin')
    dropNAN = prova.getMod('dropNAN')
    nameFeat = prova.getPrepr('nameFeat')
    numFeat = prova.getMod('numFeat')
    numLSTMLay = prova.getMod('numLSTMLay')
    numGRULay = prova.getMod('numGRULay')
    batchSize = prova.getMod('batchSize')
    units = prova.getMod('units')
    learning_rate = prova.getMod('learning_rate')
    dropRate = prova.getMod('dropRate')
    numEpoc = prova.getMod('numEpoc')
    TrainStart = prova.getConn('TrainStart')
    TrainStop = prova.getConn('TrainStop')
    TestStart = prova.getConn('TestStart')
    TestStop = prova.getConn('TestStop')
    engineId = prova.getPrepr('newEngineNm')
    timeId = prova.getPrepr('newTimeNm')
    RUL_Id =prova.getPrepr('newRULNm')
    oldNames = [nameFeat[0],'engine','time', nameFeat[1], nameFeat[2], nameFeat[3], nameFeat[4], nameFeat[5]]
    maxRUL = prova.getPrepr('maxRUL')
    filterRUL = prova.getPrepr('threshRUL')
    percentOfSplit = prova.getPrepr('percentOfSplit') / 100
    savedWindow = prova.getPrepr('savedWindow')
    size = prova.getPrepr('size')
    step = prova.getPrepr('step')
    numFilter = prova.getPrepr('numFilter')
    modelSize = size // numFilter
    possConfig = CreatePossConfig(numLSTMLay, numGRULay, units, batchSize, dropRate)
    numConfigs = len(configurations(possConfig))
    print('CONCLUSA')
    send_notification('Inizio del test rifetito alla folder ' + str(FldNm))
    
    #### CREATE FOLDER TO SAVE ####
    print('CREAZIONE CARTELLA DI SALVATAGGIO:')
    path = createFld(MthPth,FldNm)
    print('CONCLUSA')
    
    read = 0
    if read != 0:
        #### READING FROM KIRANET DB ####
        print('LETTURA DATI DA SERVER:')
        df = readingWithInfluxDB(bucket = bucket, org = org, token = token, url = url, start = TrainStart, stop = TrainStop)
        #SAVE
        savePolarsLazyFrameCSV(df, RdNm, path)
        print('CONCLUSA')

    if read == 0:
        #### ALTERNATIVE TO READING PHASE ####
        folder = r"C:\Users\Ciro\Desktop\Kiranet"
        name = 'Database.csv'
        df = readingCSVDB(name, folder)
        print('CONCLUSA')
    
    print('PREPROCESSING:')
    #### FIRST PREPROCESSING ####
    results = Preprocessing_test(df)
    print('CONCLUSO')
    
    print('GESTIONE VARIABILI CATEGORICHE:')
    #### ENCODING CATEGORICAL VARIABLES AND FIX TIME ISSUES ####
    results, cartId = dataEncoding(results, 'descr', 'cart Id')
    new_dict = {'New Cart Id': [], 'Old Cart Id': []}
    for key, value in cartId.items():
        new_dict['New Cart Id'].append(key)
        new_dict['Old Cart Id'].append(value)
    cartId = pl.LazyFrame(new_dict)#, columns=['New Cart Id', 'Old Cart Id'])
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
    endPrepr = ti.process_time()
    
    prepr_time_elapsed = endPrepr - startPrepr
    prepr_time_elapsed = ti.strftime("%H:%M:%S", ti.gmtime(prepr_time_elapsed))

    
    print('SECONDA FASE DI PREPROCESSING:')
    dataset, cartId, flag = SecondRestruct(results, cartId, engineId, timeId, RUL_Id, oldNames, path, SvNm, cartIdNm)
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
    dataset = cart_filter(cartId, serialNum, dataset, size)

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
        
        ranges = read_ranges(testing_config_path)
        
        
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


if __name__ == '__main__':
    start = '2024-03-06T16:32:29Z'
    stop = '2024-03-06T17:22:29Z'
    serialNumb = 'IFC0000950_2023410013'
    rul, dev = testing(start, stop, serialNumb)
    print('Al 90% la R.U.L. sara', rul, ' con un errore di ', dev[0], ' secondi')
    print('Al 95% la R.U.L. sara', rul, ' con un errore di ', dev[1], ' secondi')
    print('Al 99% la R.U.L. sara', rul, ' con un errore di ', dev[2], ' secondi')