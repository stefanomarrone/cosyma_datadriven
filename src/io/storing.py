

class TrainedModel():
    def __init__(self, trained_tuple):
        self.model = trained_tuple[0]
        self.ranges = trained_tuple[2]


    def serialize(self):
        retval = bytes('message', 'utf-8')
        #todo: completare il metodo
        return retval
