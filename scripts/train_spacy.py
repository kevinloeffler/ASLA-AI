from lib.ner.models.spacy_model import SpacyModel


def train_spacy(config):
    base_model = 'models/ner/spacy_0'
    path_to_data = config['paths']['data'] + 'ner/manual_training_data/generated_historic/historic_data.csv'
    path_to_safe = config['paths']['models'] + 'ner/spacy'
    iterations = 1

    model = SpacyModel(base_model)
    model.train(iterations=iterations, with_training_csv=path_to_data, safe_to=path_to_safe)
