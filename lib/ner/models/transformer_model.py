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
        # self.labels = ['O', 'PER', 'LOC']
        self.labels = ['O', 'CLT', 'LOC', 'MST', 'CLOC', 'DATE']

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

    def train(self, with_training_csv: str, safe_to: str, delimiter: str = ','):
        data = self.load_data(with_training_csv, delimiter)
        print(f'start training with {len(data)} datapoints')
        print(data[0: 10])
        self.model.train_model(train_data=data, output_dir=safe_to, show_running_loss=True)

    def predict(self, fragment: Fragment) -> PredictionResult:
        prediction, outputs = self.model.predict([fragment.text])
        print('Prediction:', prediction)
        print('Outputs:', outputs)
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

        # TODO: make accuracy method generic to the type of entities
        entities = [e for e in EntityLabel.__members__]

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

    def load_data(self, from_csv: str, delimiter: str = ',') -> pd.DataFrame:
        # Ignore
        ignored_chars = [',', '.', ';', ':']

        # Read CSV file into a pandas DataFrame
        df = pd.read_csv(from_csv, delimiter=delimiter)

        # Initialize lists to store CoNLL format data
        sentence_ids = []
        words = []
        labels = []

        # Iterate through rows and extract information
        for index, row in df.iterrows():
            sentence_id = index
            sentence = row['sentence']
            clt = row['CLT']
            loc = row['LOC']
            mst = row['MST']
            cloc = row['CLOC']
            date = row['DATE']

            # Tokenize the sentence
            sentence_tokens = sentence.split()

            # Append token-level information to lists
            for word in sentence_tokens:
                sentence_ids.append(sentence_id)
                words.append(word)
                # Check for NaN values before comparing labels
                if pd.notna(loc) and self._word_in_sentence(word, loc, ignored_chars):
                    labels.append('LOC')
                elif pd.notna(clt) and self._word_in_sentence(word, clt, ignored_chars):
                    labels.append('CLT')
                elif pd.notna(mst) and self._word_in_sentence(word, mst, ignored_chars):
                    labels.append('MST')
                elif pd.notna(cloc) and self._word_in_sentence(word, cloc, ignored_chars):
                    labels.append('CLOC')
                elif pd.notna(date) and self._word_in_sentence(word, date, ignored_chars):
                    labels.append('DATE')
                else:
                    labels.append('O')

        # Create a new DataFrame in CoNLL format
        conll_df = pd.DataFrame({
            'sentence_id': sentence_ids,
            'words': words,
            'labels': labels
        })

        return conll_df

    @staticmethod
    def _word_in_sentence(word: str, sentence: str, ignored_chars: list[str]) -> bool:
        for char in ignored_chars:
            word = word.replace(char, '')
        for s in sentence.split():
            for char in ignored_chars:
                s = s.replace(char, '')
            if word == s:
                return True

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

