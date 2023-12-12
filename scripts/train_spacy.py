from lib.ner.models.spacy_model import SpacyModel


def train_spacy(config):
    base_model = config['paths']['models'] + 'ner/de_core_news_md/de_core_news_md-3.6.0'
    path_to_data = config['paths']['data'] + 'ner/training_set_200.csv'
    path_to_safe = config['paths']['models'] + 'ner/spacy'
    iterations = 100

    model = SpacyModel(base_model)
    model.train(iterations=iterations, with_training_csv=path_to_data, safe_to=path_to_safe, gpu_id=config['gpu_id'])
