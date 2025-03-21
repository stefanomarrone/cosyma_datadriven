

class PredictionResults():

    def more_than_test(x):
        return x >= 3 * 3600

    def between_in_test(x):
        return 3600 <= x < 3 * 3600

    def less_than_test(x):
        return 3600 > x

    lookup_table = {
        'more-than-3-hours': more_than_test,
        'between-3-hours-and-1-hour': between_in_test,
        'less-than-1-hour': less_than_test
    }

    def __init__(self, predicted_rul, training_results):
        self.dictionary = dict()
        labels = list(PredictionResults.lookup_table.keys())
        flag = False
        counter = 0
        while not flag and (counter < len(labels)):
            label = labels[counter]
            flag = PredictionResults.lookup_table[label](predicted_rul)
            if not flag:
                counter += 1
        if flag:
            self.dictionary['rul'] = predicted_rul
            self.dictionary['deviations'] = training_results.getRow(labels[counter])
            for key in self.dictionary['deviations'].keys():
                temp = list(self.dictionary['deviations'][key])
                temp = list(map(lambda x: predicted_rul * x, temp))
                self.dictionary['deviations'][key] = list(temp)

    def getDictionary(self):
        return self.dictionary


class TrainingResults():
    def __init__(self, ranges):
        self.dictionary = dict()
        labels = ['more-than-3-hours', 'between-3-hours-and-1-hour', 'less-than-1-hour']
        for label, range in zip(labels,ranges):
            self.add(label, range[0.1], range[0.05], range[0.01])

    def getRow(self, label):
        return self.dictionary[label]

    def add(self, label, ninety, ninetyfive, ninetynine):
        temporary_dictionary = {
            'ninety': ninety,
            'ninetyfive': ninetyfive,
            'ninetynine': ninetynine
        }
        self.dictionary[label] = temporary_dictionary
