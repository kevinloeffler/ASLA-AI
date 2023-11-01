from lib.ner.models.transformer_model import TransformerModel


def train_transformer_model(config):
    model = TransformerModel(model_type='roberta', model_name='roberta-base', numbers_of_gpus=config['number_of_gpus'])
    model.train(with_training_csv='data/ner/manual_training_data/per_loc_1.csv',
                safe_to=config['paths']['models'] + 'ner/transformers')

