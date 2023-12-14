from lib.ner.models.transformer_model import TransformerModel


def train_transformer_model(config):

    safe_to = config['paths']['models'] + 'ner/xlm-roberta-1'

    model = TransformerModel(model_type='xlm-roberta',
                             model_name='xlm-roberta-base',  # config['paths']['models'] + 'ner/trf_bert_1',
                             training_iterations=70,
                             safe_to=safe_to,
                             numbers_of_gpus=config['number_of_gpus'],
                             gpu_id=config['gpu_id'],)
    model.train(with_training_csv=config['paths']['data'] + 'ner/training_set_200.csv',
                safe_to=safe_to,
                delimiter='\t')

# BASE MODELS:
# roberta - roberta-base
# bert - bert-base-uncased
