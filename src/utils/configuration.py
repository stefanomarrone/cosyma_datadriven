from src.utils.metaclasses import Singleton
from configparser import ConfigParser


class Configuration(metaclass=Singleton):

    def __init__(self, inifilename):
        self.board = dict()
        self.load(inifilename)

    def get(self, key):
        return self.board[key]

    def put(self, key, value):
        self.board[key] = value

    def load(self, inifile):
        reader = ConfigParser()
        try:
            reader.read(inifile)
            # section - mongodb
            temp = reader['mongodb']['address']
            self.put('mongo_address', temp)
            temp = reader['mongodb']['port']
            self.put('mongo_port', int(temp))
            # section - server
            temp = reader['server']['applicationport']
            self.put('applicationport', int(temp))
            temp = reader['server']['applicationaddress']
            self.put('applicationip', temp)
            # section - influxdb
            temp = reader['influxdb']['url']
            self.put('influx_url', temp)
            temp = reader['influxdb']['token']
            self.put('influx_tok', temp)
            temp = reader['influxdb']['org']
            self.put('influx_org', temp)
            temp = reader['influxdb']['bucket']
            self.put('influx_bucket', temp)
            # section - model
            temp = reader['model']['numlstmlay']
            self.put('numlstmlay', int(temp))
            temp = reader['model']['rangeinmin']
            self.put('rangeinmin', int(temp))
            temp = reader['model']['dropnan']
            self.put('dropnan', bool(temp))
            temp = reader['model']['numgrulay']
            self.put('numGRULay', int(temp))
            temp = reader['model']['batchsize']
            self.put('batchSize', int(temp))
            temp = reader['model']['droprate']
            self.put('dropRate', float(temp))
            temp = reader['model']['units']
            self.put('units', int(temp))
            temp = reader['model']['numepoc']
            self.put('numEpoc', int(temp))
            temp = reader['model']['learning_rate']
            self.put('learning_rate', float(temp))
            temp = reader['model']['numfeat']
            self.put('numFeat', int(temp))
            # section - preprocessing
            temp = reader['preprocessing']['namefeat']
            temp = temp.split(',')
            self.put('namefeat', temp)
            temp = reader['preprocessing']['newrulnm']
            self.put('newRULNm', temp)
            temp = reader['preprocessing']['newenginenm']
            self.put('newEngineNm', temp)
            temp = reader['preprocessing']['newtimenm']
            self.put('newTimeNm', temp)
            temp = reader['preprocessing']['maxrul']
            self.put('maxRUL', int(temp))
            temp = reader['preprocessing']['threshrul']
            self.put('threshRUL', int(temp))
            temp = reader['preprocessing']['percentofsplit']
            self.put('percentOfSplit', int(temp))
            temp = reader['preprocessing']['savedwindow']
            self.put('savedWindow', int(temp))
            temp = reader['preprocessing']['size']
            self.put('size', int(temp))
            temp = reader['preprocessing']['step']
            self.put('step', int(temp))
            temp = reader['preprocessing']['numfilter']
            self.put('numFilter', int(temp))
        except Exception as s:
            print(s)

    def __str__(self):
        return str(self.board)
