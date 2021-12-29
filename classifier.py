from sklearn import svm


CLASSIFIERS = {
    'SVC': svm.SVC(C=1, kernel='linear', decision_function_shape='ovo')
}

class DiplomacyMessageClassifier:
    def __init__(self, classifier='SVC'):
        print('-- DiplomacyMessageClassifier init --')
        self.classifier = CLASSIFIERS[classifier]
        self.name = classifier
        self.predictions = None

    def train(self, messages, labels):
        print('Training in progress...')
        self.classifier.fit(messages, labels)
        return self.classifier

    def predict(self, test_messages):
        result = self.classifier.predict(test_messages)
        self.predictions = [bool(p) for p in list(result)]
        return self.predictions
