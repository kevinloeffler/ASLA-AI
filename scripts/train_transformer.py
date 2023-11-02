from lib.ner.models.transformer_model import TransformerModel


def train_transformer_model(config):
    model = TransformerModel(model_type='bert',
                             model_name='herokiller/german-bert-finetuned-ner',  # config['paths']['models'] + 'ner/transformers_2',
                             numbers_of_gpus=config['number_of_gpus'],
                             training_iterations=200,
                             gpu_id=config['gpu_id'])
    model.train(with_training_csv=config['paths']['data'] + 'ner/manual_training_data/per_loc_1.csv',  # 'ner/manual_training_data/generated_historic/historic_data.csv'
                safe_to=config['paths']['models'] + 'ner/trf_bert_1')
