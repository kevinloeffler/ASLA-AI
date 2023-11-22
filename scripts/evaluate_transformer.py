from lib.ner.architecture import safe_predictions_to_csv
from lib.ner.models.transformer_model import TransformerModel


def evaluate_transformer_model(config):
    model = TransformerModel(model_type='roberta',
                             model_name=config['paths']['models'] + 'ner/trf_roberta_3',
                             numbers_of_gpus=config['number_of_gpus'],
                             training_iterations=1,
                             gpu_id=config['gpu_id'])
    performance = model.test(config['paths']['data'] + 'ner/manual_training_data/per_loc_validation_1.csv',
                             output_file='model_evaluation/ner/trf_roberta_3.txt')
    safe_predictions_to_csv(to='model_evaluation/ner/trf_roberta_3.csv', prediction_results=performance)

    # layout_model = TransformerModel(model_type='roberta', model_name='roberta-base')
    # layout_model = TransformerModel(model_type='bert', model_name='domischwimmbeck/bert-base-german-cased-fine-tuned-ner')
    # layout_model.train(with_training_csv='data/ner/manual_training_data/per_loc_1.csv', safe_to=config['paths']['models'] + 'ner/transformers')
    # layout_model.predict(Fragment('Garten des Herrn Gretsch', entities=[]))
