import os
import random

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import matplotlib.pyplot as plt

from lib.ner.architecture import AbstractModel, Fragment, PredictionResult, EntityLabel, Entity
from lib.ner.data import load_data


class SpacyModel(AbstractModel):

    model_name: str

    def __init__(self, model_name: str = 'de_core_news_sm'):
        self.model_name = model_name

    ########## TRAINING ##########

    def train(self, iterations: int, with_training_csv: str, safe_to: str = '') -> list:
        print(f'Loading model: "{self.model_name}"')
        spacy.require_gpu()
        model = spacy.load(self.model_name)
        pipeline = model.get_pipe('ner')

        training_data = load_data(from_csv=with_training_csv, skip_overlapping=True)
        training_data = self.__convert_training_data(training_data)
        print('Start training spacy model with:', len(training_data), 'datapoints')

        for label in EntityLabel.__members__:
            pipeline.add_label(label)

        # Disable pipeline components we dont want to change
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        unaffected_pipes = [pipe for pipe in model.pipe_names if pipe not in pipe_exceptions]

        losses = []

        # train
        with model.disable_pipes(unaffected_pipes):

            for iteration in range(iterations):
                # randomize training data
                random.shuffle(training_data)
                loss = {}

                # creating batches
                batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    for text, annotations in batch:
                        doc = model.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        model.update([example], drop=0.35, losses=loss)

                losses.append(loss['ner'])
                print(f'Iteration {iteration + 1}/{iterations}: Losses', loss)

        print('Saving model to disk...')
        self.__safe_model(model=model, path=safe_to)

        plt.plot(losses)
        plt.title('Spacy model training loss')
        plt.show()

        return losses

    ########## TESTING ##########

    def predict(self, model, fragment: Fragment) -> PredictionResult:
        datapoint = self.convert_fragment_to_data(fragment)
        prediction = model(datapoint[0])
        predicted_entities = [self.__convert_model_prediction_to_entity(p, fragment) for p in prediction.ents]
        return self.evaluate_prediction(fragment=fragment, predicted_entities=predicted_entities)

    def test(self, with_testing_csv: str) -> list[PredictionResult]:
        model = self.get_model()
        results = []

        data = load_data(with_testing_csv)
        for datapoint in data:
            results.append(self.predict(model, datapoint))

        model_accuracy = sum([p.accuracy if p.accuracy else 0 for p in results]) / len(results)

        per_count, loc_count = 0, 0
        per_total_accuracy, loc_total_accuracy = 0, 0

        for result in results:
            if EntityLabel.PER.name in result.entity_accuracy:
                per_count += 1
                per_total_accuracy += result.entity_accuracy[EntityLabel.PER.name]
            if EntityLabel.LOC.name in result.entity_accuracy:
                loc_count += 1
                loc_total_accuracy += result.entity_accuracy[EntityLabel.LOC.name]

        print('Model accuracy:', model_accuracy)
        print('           PER:', per_total_accuracy / per_count)
        print('           LOC:', loc_total_accuracy / loc_count)

        return results

    ########## UTIL ##########

    def __convert_training_data(self, data: list[Fragment]) -> list:
        non_overlapping_fragments = list(filter(lambda f: not f.entities_overlap(), data))
        return [self.convert_fragment_to_data(f) for f in non_overlapping_fragments]

    def ctd(self, data) -> list:
        return self.__convert_training_data(data=data)

    @staticmethod
    def __convert_model_prediction_to_entity(prediction, fragment: Fragment) -> Entity:
        return Entity(text=prediction.text, label=EntityLabel[prediction.label_], title=fragment.text)

    @staticmethod
    def convert_fragment_to_data(fragment: Fragment) -> any:
        entities = [(e.start_index, e.end_index, e.label.name) for e in fragment.entities]
        return fragment.text, {'entities': entities}

    @staticmethod
    def __safe_model(model, path: str):
        path = path if path != '' else 'models/ner/spacy'

        suffix = 0
        while os.path.exists(path + '_' + str(suffix)):
            suffix += 1
        model.to_disk(path + '_' + str(suffix))

    def get_model(self):
        print(f'Loading model: "{self.model_name}"')
        return spacy.load(self.model_name)
