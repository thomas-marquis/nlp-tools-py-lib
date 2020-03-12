import typing as t
import numpy as np


class AbstractClassifier:
    def build(self, data):
        raise Exception('build method not implemented')

    def train(self):
        raise Exception('train method not implemented')

    def predict(self, value):
        raise Exception('predict method not implemented')


class ClassificationProcessor:
    training_set: t.Dict[str, np.array]
    classifier: AbstractClassifier

    def __init__(self,
                 classifier: AbstractClassifier,
                 training_set: t.Dict[str, np.array]):
        self.training_set = training_set
        self.classifier = classifier
        self.classifier.build(training_set)

    def train(self):
        self.classifier.train()

    def predict(self, data: np.array):
        return self.classifier.predict(data)


class NaiveBayseClassifier(AbstractClassifier):
    """besoin de la représentation mergée des exemples
    """
    data: t.Dict[str, np.array]
    proba_per_intent: t.Dict[str, np.array]

    def build(self, data: t.Dict[str, np.array]):
        self.data = data

    def get_proba_per_intent(self, examples):
        return sum(list(examples)) / len(examples)

    def train(self):
        self.proba_per_intent: t.Dict[str, np.array] = \
            {k: self.get_proba_per_intent(self.data[k]) for k in self.data}

    def predict(self, value: np.array) -> t.Tuple[str, float]:
        max_p = 0
        intents = list(self.data.keys())
        best_intent = 'not_found'
        for intent in intents:
            repres = value * self.proba_per_intent[intent]
            repres = [e for e in repres if e != 0]
            current_p = 0
            if len(repres) != 0:
                current_p = np.sum(repres)
            if current_p > max_p:
                max_p = current_p
                best_intent = intent
        return best_intent, max_p


class NaiveBayseTfIdfClassifier(NaiveBayseClassifier):
    documents_per_intent: t.Dict[str, int]

    def build(self, data):
        super().build(data)
        self.documents_per_intent = {k: len(self.data[k]) for k in self.data}

    def count_unique_occurences(self, example: np.array) -> np.array:
        return [1 if occ != 0 else 0 for occ in example]

    def count_occurences_for_examples(self, examples: np.array) -> np.array:
        return np.array([self.count_unique_occurences(ex) for ex in examples])

    def get_unique_occurence_by_intent(self) -> t.Dict[str, np.array]:
        return {k: sum(self.count_occurences_for_examples(self.data[k])) for k in self.data}

    def train(self):
        get_tf = lambda examples: self.get_proba_per_intent(examples)
        tf = {k: get_tf(self.data[k]) for k in self.data}

        term_unique_apparition = self.get_unique_occurence_by_intent()
        occurences_counts: np.array = \
            sum([term_unique_apparition[k] for k in term_unique_apparition])

        documents_nb: int = sum(
            [self.documents_per_intent[k] for k in self.documents_per_intent])
        idf = np.log10(
            np.divide(
                np.full(len(occurences_counts), documents_nb),
                occurences_counts))

        self.tf_idf = {k: np.multiply(tf[k], idf) for k in tf}

    def predict(self, value: np.array) -> t.Tuple[str, float]:
        max_p = 0
        intents = list(self.data.keys())
        best_intent = 'not_found'
        for intent in intents:
            repres = value * self.tf_idf[intent]
            repres = [e for e in repres if e != 0]
            current_p = 0
            if len(repres) != 0:
                current_p = np.sum(repres)
            if current_p > max_p:
                max_p = current_p
                best_intent = intent
        return best_intent, max_p
