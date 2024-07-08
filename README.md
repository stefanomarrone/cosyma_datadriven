# cosyma_datadriven

To run the service: python3 main.py config.ini

spiegare la sintassi dei parametri passati sia al servizio di train che di testing
esempio questi sono i dati come ci si aspetta che vengano forniti **training**
startdate = 2024-03-04T16:32:29Z
stopdate = 2024-03-06T16:32:29Z


#da tradurre
Nella cartella vi sono 6 file:

- functions.py : è lo script con tutte le funzioni definite per eseguire sia la fase di Training che la fase di Testing dei carrelli. E' divisa in diverse sezioni ognuna delle quali descrive una specifica fase di preprocessing o di modellazione dell'algoritmo predittore.

- Training.py : è lo script che va lanciato per effettuare l'addestramento del modello predittore. E' basato su un piccolo tuning compiuto sia sulla struttura del modello, su alcuni parametri specifici del problema preso in oggetto, su alcuni iperparametri specifici dei modelli di reti neurali convoluzionali. E' un tuning che si potrebbe rendere leggermente più efficace aumentando il range di variabilità dei parametri presi in esame, ed aggiungendo alla lista dei parametri da monitorare alcuni parametri altrettanto sensibili.
La funzione prende in ingresso due valori, che indicano l'istante iniziale e finale in cui eseguire la query che raccoglierà i dati da preprocessare e poi da fornire in input al modello per poter essere addestrato; in output fornisce un percorso che indica il path del modello che ha ottenuto una più elevata accuratezza in termini di R.M.S.E.
NOTA BENE: lo stesso output viene salvato in un file di configurazione per la fase di Testing, dunque nella successiva fase è automaticamente caricato da esso.

- Testing.py : è lo script che viene lanciato per effettuare il testing dei dati. In input ha analogamente al train due istanti temporali che indicano la data di inizio e di fine della query compiuta su influxDB per raccogliere i dati, inoltre avrà il seriale del carrello da monitorare. In output fornirà la R.U.L. del carrello scelto, misurata in secondi, con tre rispettivi intervalli di confidenza, che vanno a fornire una affidabilità del 90%, 95% e 99% del risultato ottenuto.

- Training.ini : è il file di configurazione utilizzato nella fase di training. Qui vengono salvati gli istanti temporali in cui viene eseguita la query su influxDB forniti in input tramite la funzione Training.py.

- Testing.ini : è il file di configurazione utilizzato nella fase di training per essere aggiornato, e di testing per raccogliere le informazioni necessarie per effettuare la predizione. In fase di training nel file vengono aggiornati sia il path del modello migliore, che i valori descriventi i diversi intervalli di confidenza della predizione; mentre in fase di testing viene richiamato per capire quale motore monitorare (al suo interno è presente il seriale del motore).

- config.ini : è il file di configurazione principale. E' un file che non interessa agli utilizzatori del servizio, ma che può servire agli sviluppatori per monitorare alcuni parametri sensibili sia specifici del problema preso in analisi, sia più generici, riguardanti i percorsi in cui salvare dati e grafici inerenti le diverse fasi.

-Train - Sample View.csv e CartId.csv, due file utilizzati in alternativa al download dei dati tramite connessione VPN con influxDB. NOTA BENE: inserire i due file all'interno della cartella comparente nel percorso mthpth specificato nel file di configurazione 'config.ini'.

- requirements.txt: un file contenente le librerie necessarie per l'esecuzione di training e testing.


Per poter utilizzare i dati riassumendo basta:
1 - scaricare tutti i file presenti in questa cartella;
2 - installare i pacchetti necessari dal file requirements.txt;
2 - specificare nel file 'config.ini' il percorso in cui si trovano i due file .csv (ovvero modificare il valore della variabile mthpth);
3 - lanciare prima il file di training (Training.py) in modo da creare dei modelli e salvare in particolare il modello con migliori prestazioni;
4 - lanciare il file di testing (Testing.py).

