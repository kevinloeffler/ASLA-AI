import csv
import os
import re

import pandas as pd
import torch
from simpletransformers.ner import NERModel, NERArgs

from lib.ner.architecture import Fragment, PredictionResult, Entity, EntityLabel, evaluate_prediction
from lib.ner.data import load_data


class TransformerModel:

    def __init__(self, model_type: str, model_name: str, numbers_of_gpus: int, training_iterations: int, gpu_id=4):
        self.__has_cuda = torch.cuda.is_available()
        print('CUDA enabled:', self.__has_cuda)
        use_cuda = self.__has_cuda

        # labels = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        self.labels = ['O', 'PER', 'LOC']

        model_args = NERArgs()
        model_args.labels_list = self.labels
        model_args.num_train_epochs = training_iterations
        model_args.use_multiprocessing = True
        model_args.save_model_every_epoch = False
        model_args.wandb_project = 'asla-ai'
        if numbers_of_gpus > 0:
            use_cuda = True
            model_args.n_gpu = numbers_of_gpus
            print(f'using {numbers_of_gpus} GPUs')

        self.model = NERModel(model_type, model_name, use_cuda=use_cuda, cuda_device=gpu_id, args=model_args)

    def train(self, with_training_csv: str, safe_to: str):
        data = self.load_data(with_training_csv)
        self.model.train_model(train_data=data, output_dir=safe_to, show_running_loss=True)

    def predict(self, fragment: Fragment) -> PredictionResult:
        prediction, outputs = self.model.predict([fragment.text])
        return self.evaluate_model_prediction(fragment, prediction)

    def test(self, with_testing_csv: str, output_file: str) -> list[PredictionResult]:
        if os.path.exists(output_file):
            print('ERROR: Output file already exists')
            return None

        results = []
        data = load_data(with_testing_csv)
        for datapoint in data:
            results.append(self.predict(datapoint))

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

        with open(output_file, 'x') as file:
            file.write(f'Model accuracy: {model_accuracy}\n')
            file.write(f'           PER: {per_total_accuracy / per_count}\n')
            file.write(f'           LOC: {loc_total_accuracy / loc_count}\n')

        return results

    def evaluate(self, with_testing_csv: str, safe_to: str):
        data = self.load_data(with_testing_csv)
        self.model.eval_model(data, output_dir=safe_to)

    ########## UTIL ##########

    @staticmethod
    def load_data(from_csv: str) -> pd.DataFrame:
        data = {
            'sentence_id': [],
            'words': [],
            'labels': [],
        }

        chars_to_ignore = ',.:'

        with open(from_csv, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)

            for index, row in enumerate(reader):
                sentence_tokenized: list[str] = row[0].split(' ')
                data['sentence_id'] += [index] * len(sentence_tokenized)
                data['words'] += sentence_tokenized

                labels = []
                for token_index, token in enumerate(sentence_tokenized):
                    local_labels = []
                    for label_index, label in enumerate(header[1:]):
                        if re.sub(f'[{chars_to_ignore}]', '', token) in row[label_index + 1].split(' '):
                            local_labels.append(label)
                    if len(local_labels) == 1:
                        labels.append(local_labels[0])
                    else:
                        labels.append('O')

                data['labels'] += labels

        return pd.DataFrame(data=data)

    @staticmethod
    def evaluate_model_prediction(fragment: Fragment, prediction: list[list[dict[str, str]]]) -> PredictionResult:
        prediction = prediction[0]
        predicted_entities = []

        for token in prediction:
            word = list(token.keys())[0]
            label = list(token.values())[0]
            if label in EntityLabel.__members__:
                predicted_entities.append(Entity(word, EntityLabel[label], fragment.text))

        return evaluate_prediction(fragment, predicted_entities)

