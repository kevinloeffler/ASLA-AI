import unittest

from lib.ner.architecture import EntityLabel, Fragment, Entity, evaluate_prediction_2


class TestPredictionEvaluation(unittest.TestCase):

    def test_perfect_prediction(self):
        sentence = 'Garten des Herrn Gretsch in Jona'
        fragment = Fragment(text=sentence, entities=[
            Entity(text='Herrn', label=EntityLabel.CLT, title=sentence),
            Entity(text='Gretsch', label=EntityLabel.CLT, title=sentence),
            Entity(text='Jona', label=EntityLabel.LOC, title=sentence)
        ])
        prediction = [
            Entity(text='Herrn', label=EntityLabel.CLT, title=sentence),
            Entity(text='Gretsch', label=EntityLabel.CLT, title=sentence),
            Entity(text='Jona', label=EntityLabel.LOC, title=sentence)
        ]
        prediction_result = evaluate_prediction_2(fragment=fragment, predicted_entities=prediction)
        # prediction accuracy
        assert prediction_result.accuracy == 1, f'accuracy should be 1, is {prediction_result.accuracy}'
        assert prediction_result.precision == 1, f'precision should be 1, is {prediction_result.precision}'
        assert prediction_result.recall == 1, f'recall should be 1, is {prediction_result.recall}'
        # entity accuracy
        assert prediction_result.entity_accuracy['CLT'] == 1, f'CLT accuracy should be 1, is {prediction_result.entity_accuracy["CLT"]}'
        assert prediction_result.entity_accuracy['LOC'] == 1, f'LOC accuracy should be 1, is {prediction_result.entity_accuracy["LOC"]}'

    def test_correct_empty_prediction(self):
        sentence = 'Text ohne Entities'
        fragment = Fragment(text=sentence, entities=[])
        prediction = []
        prediction_result = evaluate_prediction_2(fragment=fragment, predicted_entities=prediction)
        # prediction accuracy
        assert prediction_result.accuracy == 1, f'accuracy should be 1, is {prediction_result.accuracy}'
        assert prediction_result.precision == 1, f'precision should be 1, is {prediction_result.precision}'
        assert prediction_result.recall == 1, f'recall should be 1, is {prediction_result.recall}'
        # entity accuracy
        assert prediction_result.entity_accuracy == {}, f'entity accuracy should be empty dictionary but is {prediction_result.entity_accuracy}'


    def test_incorrect_empty_prediction(self):
        sentence = 'Text mit Entity'
        fragment = Fragment(text=sentence, entities=[Entity(text='Entity', label=EntityLabel.LOC, title=sentence)])
        prediction = []
        prediction_result = evaluate_prediction_2(fragment=fragment, predicted_entities=prediction)
        # prediction accuracy
        assert prediction_result.accuracy == 0, f'accuracy should be 0, is {prediction_result.accuracy}'
        assert prediction_result.precision == 0, f'precision should be 0, is {prediction_result.precision}'
        assert prediction_result.recall == 0, f'recall should be 0, is {prediction_result.recall}'
        # entity accuracy
        assert prediction_result.entity_accuracy['LOC'] == 0, f'LOC accuracy should be 0, is {prediction_result.entity_accuracy["LOC"]}'


    def test_false_positive(self):
        sentence = 'Text ohne Entities'
        fragment = Fragment(text=sentence, entities=[])
        prediction = [
            Entity(text='Text', label=EntityLabel.CLT, title=sentence),
            Entity(text='Entities', label=EntityLabel.LOC, title=sentence)
        ]
        prediction_result = evaluate_prediction_2(fragment=fragment, predicted_entities=prediction)
        # prediction accuracy
        assert prediction_result.accuracy == 0, f'accuracy should be 0, is {prediction_result.accuracy}'
        assert prediction_result.precision == 0, f'precision should be 0, is {prediction_result.precision}'
        assert prediction_result.recall == 0, f'recall should be 0, is {prediction_result.recall}'
        # entity accuracy
        assert prediction_result.entity_accuracy['CLT'] == 0, f'CLT accuracy should be 0, is {prediction_result.entity_accuracy["CLT"]}'
        assert prediction_result.entity_accuracy['LOC'] == 0, f'LOC accuracy should be 0, is {prediction_result.entity_accuracy["LOC"]}'


    def test_example_1(self):
        sentence = 'Garten des Herrn Gretsch in Jona'
        fragment = Fragment(text=sentence, entities=[
            Entity(text='Herrn', label=EntityLabel.CLT, title=sentence),
            Entity(text='Gretsch', label=EntityLabel.CLT, title=sentence),
            Entity(text='Jona', label=EntityLabel.LOC, title=sentence)
        ])
        prediction = [
            Entity(text='Gretsch', label=EntityLabel.CLT, title=sentence),
            Entity(text='Jona', label=EntityLabel.MST, title=sentence)
        ]
        prediction_result = evaluate_prediction_2(fragment=fragment, predicted_entities=prediction)
        # prediction accuracy
        assert prediction_result.accuracy == 0.4, f'accuracy should be 0.4, is {prediction_result.accuracy}'
        assert prediction_result.precision == 0.5, f'precision should be 0.5, is {prediction_result.precision}'
        assert prediction_result.recall == (1/3), f'recall should be 1/3, is {prediction_result.recall}'
        # entity accuracy
        assert prediction_result.entity_accuracy['CLT'] == (2/3), f'CLT accuracy should be 0.66, is {prediction_result.entity_accuracy["CLT"]}'
        assert prediction_result.entity_accuracy['LOC'] == 0, f'LOC accuracy should be 0, is {prediction_result.entity_accuracy["LOC"]}'
        assert prediction_result.entity_accuracy['MST'] == 0, f'LOC accuracy should be 0, is {prediction_result.entity_accuracy["MST"]}'

    def test_example_2(self):
        sentence = 'Detail Eisenhag + Tor 1:20 17.6.1968'
        fragment = Fragment(text=sentence, entities=[
            Entity(text='1:20', label=EntityLabel.MST, title=sentence),
            Entity(text='17.6.1968', label=EntityLabel.DATE, title=sentence),
        ])
        prediction = [
            Entity(text='1:20', label=EntityLabel.MST, title=sentence),
            Entity(text='17.6.1968', label=EntityLabel.CLOC, title=sentence),
        ]
        prediction_result = evaluate_prediction_2(fragment=fragment, predicted_entities=prediction)
        # prediction accuracy
        assert prediction_result.accuracy == 0.5, f'accuracy should be 0.5, is {prediction_result.accuracy}'
        assert prediction_result.precision == 0.5, f'precision should be 0.5, is {prediction_result.precision}'
        assert prediction_result.recall == 0.5, f'recall should be 0.5, is {prediction_result.recall}'
        # entity accuracy
        assert prediction_result.entity_accuracy['MST'] == 1, f'LOC accuracy should be 1, is {prediction_result.entity_accuracy["MST"]}'
        assert prediction_result.entity_accuracy['DATE'] == 0, f'DATE accuracy should be 0, is {prediction_result.entity_accuracy["DATE"]}'
        assert prediction_result.entity_accuracy['CLOC'] == 0, f'CLOC accuracy should be 0, is {prediction_result.entity_accuracy["CLOC"]}'
