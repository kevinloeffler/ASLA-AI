import unittest

import pandas as pd

from lib.ner.models.transformer_model import TransformerModel


class TransformerModelTest(unittest.TestCase):
    pass


class TransformerLoadData(unittest.TestCase):
    model = TransformerModel(model_name='bert-base-uncased', model_type='bert', numbers_of_gpus=0, training_iterations=1)

    def test_word_in_sentence(self):
        ignored_chars = [',', '.', ';', ':']
        sentence = 'Garten Herr Dr. Vogel-Mayer, Trogen, M: 1:100'
        assert self.model._word_in_sentence(sentence.split()[0], sentence, ignored_chars), 'Garten should be in sentence'
        assert self.model._word_in_sentence(sentence.split()[1], sentence, ignored_chars), 'Herr should be in sentence'
        assert self.model._word_in_sentence(sentence.split()[2], sentence, ignored_chars), 'Dr. should be in sentence'
        assert self.model._word_in_sentence(sentence.split()[3], sentence, ignored_chars), 'Vogel-Mayer, should be in sentence'
        assert self.model._word_in_sentence(sentence.split()[4], sentence, ignored_chars), 'Trogen, should be in sentence'
        assert self.model._word_in_sentence(sentence.split()[5], sentence, ignored_chars), 'M: should be in sentence'
        assert self.model._word_in_sentence(sentence.split()[6], sentence, ignored_chars), '1:100 should be in sentence'

    def test_load_data(self):
        input_csv = '/Users/kl/Kevin/Projects/ASLA/ASLA-AI/data/ner/test_training_set.csv'

        sentence_0 = pd.DataFrame({
            'sentence_id': [0, 0, 0, 0, 0, 0, 0],
            'words': ['Garten', 'Herr', 'Dr.', 'Vogel-Mayer,', 'Trogen,', 'M:', '1:100'],
            'labels': ['O', 'CLT', 'CLT', 'CLT', 'LOC', 'O', 'MST'],
        })
        sentence_1 = pd.DataFrame({
            'sentence_id': [1],
            'words': ['1944-12-08'],
            'labels': ['DATE'],
        })

        output = self.model.load_data(input_csv, delimiter='\t')
        pd.testing.assert_frame_equal(output[0: 7], sentence_0)
        pd.testing.assert_frame_equal(output[7: 8].reset_index(drop=True), sentence_1)


