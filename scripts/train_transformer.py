from lib.ner.models.transformer_model import TransformerModel
import wandb


def train_transformer_model(config):

    safe_to = config['paths']['models'] + 'ner/xlm-roberta-100'

    model = TransformerModel(model_type='roberta',
                             model_name='xlm-roberta-base',  # config['paths']['models'] + 'ner/trf_bert_1',
                             training_iterations=60,
                             safe_to=safe_to,
                             numbers_of_gpus=config['number_of_gpus'],
                             gpu_id=config['gpu_id'],)
    model.train(with_training_csv=config['paths']['data'] + 'ner/training_set_100.csv',
                safe_to=safe_to,
                delimiter='\t')

    wandb.finish()

    # REPEAT ---------------------------------------

    safe_to = config['paths']['models'] + 'ner/xlm-roberta-500'
    model = TransformerModel(model_type='roberta',
                             model_name='xlm-roberta-base',  # config['paths']['models'] + 'ner/trf_bert_1',
                             training_iterations=60,
                             safe_to=safe_to,
                             numbers_of_gpus=config['number_of_gpus'],
                             gpu_id=config['gpu_id'],)
    model.train(with_training_csv=config['paths']['data'] + 'ner/training_set_500.csv',
                safe_to=safe_to,
                delimiter='\t')

    wandb.finish()

    safe_to = config['paths']['models'] + 'ner/xlm-roberta-1000'
    model = TransformerModel(model_type='roberta',
                             model_name='xlm-roberta-base',  # config['paths']['models'] + 'ner/trf_bert_1',
                             training_iterations=90,
                             safe_to=safe_to,
                             numbers_of_gpus=config['number_of_gpus'],
                             gpu_id=config['gpu_id'],)
    model.train(with_training_csv=config['paths']['data'] + 'ner/training_set_1000.csv',
                safe_to=safe_to,
                delimiter='\t')

    wandb.finish()

    safe_to = config['paths']['models'] + 'ner/xlm-roberta-2000'
    model = TransformerModel(model_type='roberta',
                             model_name='xlm-roberta-base',  # config['paths']['models'] + 'ner/trf_bert_1',
                             training_iterations=120,
                             safe_to=safe_to,
                             numbers_of_gpus=config['number_of_gpus'],
                             gpu_id=config['gpu_id'],)
    model.train(with_training_csv=config['paths']['data'] + 'ner/training_set_2000.csv',
                safe_to=safe_to,
                delimiter='\t')


# BASE MODELS:
# roberta - roberta-base
# bert - bert-base-uncased
