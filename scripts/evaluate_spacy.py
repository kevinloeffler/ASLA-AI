from lib.ner.models.spacy_model import SpacyModel


def evaluate_spacy(config):
    base_model = config['paths']['models'] + 'ner/spacy_5'

    model = SpacyModel(base_model)
    model.test(with_testing_csv=config['paths']['data'] + 'ner/manual-validation-1.tsv',
               output_file=f'model_evaluation/ner/spacy.txt',
               delimiter='\t')
