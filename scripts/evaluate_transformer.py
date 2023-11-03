from lib.ner.architecture import safe_predictions_to_csv
from lib.ner.models.transformer_model import TransformerModel


def evaluate_transformer_model(config):
    model = TransformerModel(model_type='bert',
                             model_name=config['paths']['models'] + 'ner/trf_bert_2',
                             numbers_of_gpus=config['number_of_gpus'],
                             training_iterations=1,
                             gpu_id=config['gpu_id'])
    performance = model.test(config['paths']['data'] + 'ner/manual_training_data/per_loc_validation_1.csv',
                             output_file='model_evaluation/ner/trf_bert_2.txt')
    safe_predictions_to_csv(to='model_evaluation/ner/trf_bert_2.csv', prediction_results=performance)

    # model = TransformerModel(model_type='roberta', model_name='roberta-base')
    # model = TransformerModel(model_type='bert', model_name='domischwimmbeck/bert-base-german-cased-fine-tuned-ner')
    # model.train(with_training_csv='data/ner/manual_training_data/per_loc_1.csv', safe_to=config['paths']['models'] + 'ner/transformers')
    # model.predict(Fragment('Garten des Herrn Gretsch', entities=[]))
