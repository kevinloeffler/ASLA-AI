from lib.ner.models.transformer_model import TransformerModel


def train_transformer_model(config):

    safe_to = config['paths']['models'] + 'ner/roberta-large-100'

    model = TransformerModel(model_type='roberta',
                             model_name='roberta-large',  # config['paths']['models'] + 'ner/trf_bert_1',
                             training_iterations=70,
                             safe_to=safe_to,
                             numbers_of_gpus=config['number_of_gpus'],
                             gpu_id=config['gpu_id'],)
    model.train(with_training_csv=config['paths']['data'] + 'ner/training_set_100.csv',
                safe_to=safe_to,
                delimiter='\t')

    # REPEAT ---------------------------------------

    safe_to = config['paths']['models'] + 'ner/roberta-large-500'
    model = TransformerModel(model_type='roberta',
                             model_name='roberta-large',  # config['paths']['models'] + 'ner/trf_bert_1',
                             training_iterations=70,
                             safe_to=safe_to,
                             numbers_of_gpus=config['number_of_gpus'],
                             gpu_id=config['gpu_id'],)
    model.train(with_training_csv=config['paths']['data'] + 'ner/training_set_500.csv',
                safe_to=safe_to,
                delimiter='\t')


    safe_to = config['paths']['models'] + 'ner/roberta-large-1000'
    model = TransformerModel(model_type='roberta',
                             model_name='roberta-large',  # config['paths']['models'] + 'ner/trf_bert_1',
                             training_iterations=70,
                             safe_to=safe_to,
                             numbers_of_gpus=config['number_of_gpus'],
                             gpu_id=config['gpu_id'],)
    model.train(with_training_csv=config['paths']['data'] + 'ner/training_set_1000.csv',
                safe_to=safe_to,
                delimiter='\t')


    safe_to = config['paths']['models'] + 'ner/roberta-large-2000'
    model = TransformerModel(model_type='roberta',
                             model_name='roberta-large',  # config['paths']['models'] + 'ner/trf_bert_1',
                             training_iterations=70,
                             safe_to=safe_to,
                             numbers_of_gpus=config['number_of_gpus'],
                             gpu_id=config['gpu_id'],)
    model.train(with_training_csv=config['paths']['data'] + 'ner/training_set_2000.csv',
                safe_to=safe_to,
                delimiter='\t')


# BASE MODELS:
# roberta - roberta-base
# bert - bert-base-uncased
