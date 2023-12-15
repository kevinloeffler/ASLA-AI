from lib.ner.architecture import safe_predictions_to_csv
from lib.ner.models.transformer_model import TransformerModel


def evaluate_transformer_model(config):

    model_name = 'xlm-roberta-1'

    model = TransformerModel(model_type='roberta',
                             model_name=config['paths']['models'] + 'ner/' + model_name,
                             training_iterations=1,
                             safe_to=config['paths']['models'] + 'ner/' + model_name,
                             numbers_of_gpus=config['number_of_gpus'],
                             gpu_id=config['gpu_id'])

    performance = model.test(config['paths']['data'] + 'ner/manual-validation-1.tsv',
                             output_file=f'model_evaluation/ner/{model_name}.txt',
                             delimiter='\t')

    safe_predictions_to_csv(to=f'model_evaluation/ner/{model_name}.csv', prediction_results=performance)

    # REPEAT ------------------------------------

    model_name = 'german-bert-1'

    model = TransformerModel(model_type='bert',
                             model_name=config['paths']['models'] + 'ner/' + model_name,
                             training_iterations=1,
                             safe_to=config['paths']['models'] + 'ner/' + model_name,
                             numbers_of_gpus=config['number_of_gpus'],
                             gpu_id=config['gpu_id'])

    performance = model.test(config['paths']['data'] + 'ner/manual-validation-1.tsv',
                             output_file=f'model_evaluation/ner/{model_name}.txt',
                             delimiter='\t')

    safe_predictions_to_csv(to=f'model_evaluation/ner/{model_name}.csv', prediction_results=performance)

    model_name = 'roberta-large-1'

    model = TransformerModel(model_type='roberta',
                             model_name=config['paths']['models'] + 'ner/' + model_name,
                             training_iterations=1,
                             safe_to=config['paths']['models'] + 'ner/' + model_name,
                             numbers_of_gpus=config['number_of_gpus'],
                             gpu_id=config['gpu_id'])

    performance = model.test(config['paths']['data'] + 'ner/manual-validation-1.tsv',
                             output_file=f'model_evaluation/ner/{model_name}.txt',
                             delimiter='\t')

    safe_predictions_to_csv(to=f'model_evaluation/ner/{model_name}.csv', prediction_results=performance)



    # layout_model = TransformerModel(model_type='roberta', model_name='roberta-base')
    # layout_model = TransformerModel(model_type='bert', model_name='domischwimmbeck/bert-base-german-cased-fine-tuned-ner')
    # layout_model.train(with_training_csv='data/ner/manual_training_data/per_loc_1.csv', safe_to=config['paths']['models'] + 'ner/transformers')
    # layout_model.predict(Fragment('Garten des Herrn Gretsch', entities=[]))

# BASE MODELS:
# roberta - roberta-base
# bert - bert-base-uncased
