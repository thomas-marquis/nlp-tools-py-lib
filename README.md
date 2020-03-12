# nlp-tools-py-lib
python simple nlp library

## installation

````shell script
pip install nlp-tools-py-lib
````

## usage

````python
# main.py
from nlp_tools.preprocessing import Preprocessing
from nlp_tools.loaders import MdLoader
from nlp_tools.representations import MergedMatrixRepresentation
from nlp_tools.classifiers import ClassificationProcessor, NaiveBayseTfIdfClassifier

TRAIN_PATH = 'demo_training.md'

def build_classifier():
    loader = MdLoader(TRAIN_PATH)
    processor = Preprocessing(loader)
    repres = MergedMatrixRepresentation(processor.data)
    classifier = ClassificationProcessor(NaiveBayseTfIdfClassifier(), repres.data)
    classifier.train()

    def predict(text: str):
        message = repres.process_new_data(processor.process_sentence(text))
        intent, score = classifier.predict(message)
        return intent, score
    return predict
````

``training.md`` example :

````markdown
# intents

## my_first_intent_name

### responses

- ...

### example

- ...
````