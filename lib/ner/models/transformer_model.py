import csv
import re

import pandas as pd


class TransformerModel:

    # labels = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    labels = ['PER', 'LOC']

    # cuda_is_available = torch.cuda.is_available()
    #model = NERModel('bert',
    #                 'domischwimmbeck/bert-base-german-cased-fine-tuned-ner',
    #                 use_cuda=cuda_is_available)

    ########## UTIL ##########

    @staticmethod
    def load_data(from_csv: str) -> pd.DataFrame:
        data = {
            'sentence_index': [],
            'words': [],
            'labels': [],
        }

        chars_to_ignore = ',.:'

        with open(from_csv, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            print('header:', header)

            for index, row in enumerate(reader):
                sentence_tokenized: list[str] = row[0].split(' ')
                data['sentence_index'] += [index] * len(sentence_tokenized)
                data['words'] += sentence_tokenized

                labels = []
                for token_index, token in enumerate(sentence_tokenized):
                    local_labels = []
                    for label_index, label in enumerate(header[1:]):
                        if re.sub(f'[{chars_to_ignore}]', '', token) in row[label_index + 1].split(' '):
                            local_labels.append(label)
                    if not local_labels:
                        labels.append('O')
                    else:
                        labels.append(' '.join(local_labels))

                data['labels'] += labels
        return pd.DataFrame(data=data)

