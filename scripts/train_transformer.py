from lib.ner.models.transformer_model import TransformerModel


def train_transformer_model(config):
    model = TransformerModel(model_type='bert',
                             model_name='bert-base-uncased',  # config['paths']['models'] + 'ner/trf_bert_1',
                             numbers_of_gpus=config['number_of_gpus'],
                             training_iterations=100,
                             gpu_id=config['gpu_id'])
    model.train(with_training_csv=config['paths']['data'] + 'ner/training_set_250.csv',
                safe_to=config['paths']['models'] + 'ner/trf_bert_4')

# BASE MODELS:
# roberta - roberta-base
# bert - bert-base-uncased
