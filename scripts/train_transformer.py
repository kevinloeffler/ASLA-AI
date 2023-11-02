from lib.ner.models.transformer_model import TransformerModel


def train_transformer_model(config):
    model = TransformerModel(model_type='roberta',
                             model_name=config['paths']['models'] + 'ner/transformers_2',
                             numbers_of_gpus=config['number_of_gpus'],
                             training_iterations=250)
    model.train(with_training_csv=config['paths']['data'] + 'ner/manual_training_data/generated_historic/historic_data.csv',
                safe_to=config['paths']['models'] + 'ner/transformers')
