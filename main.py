import tomli

from scripts.evaluate_transformer import evaluate_transformer_model
from scripts.train_transformer import train_transformer_model

if __name__ == '__main__':
    # config = None
    with open('config.toml', mode='rb') as config_file:
        config = tomli.load(config_file)
        try:
            config['paths']['data']
        except KeyError:
            print('ERROR: missing config key: paths.data')
            quit(1)
        try:
            config['paths']['models']
        except KeyError:
            print('ERROR: missing config key: paths.models')
            quit(1)
        try:
            config['number_of_gpus']
        except KeyError:
            print('ERROR: missing config key: number_of_gpus')

    evaluate_transformer_model(config)
    # train_transformer_model(config)

# data = model.load_data('data/ner/manual_training_data/per_loc_1.csv')
# print(data[200:250])

# spacy_model = SpacyModel(model_name='de_core_news_sm')
# test_fragment = load_data('manual_training_data/manual_training_data_1.csv')[345]
# print('test fragment:', test_fragment)
# prediction = spacy_model.predict(spacy_model.get_model(), test_fragment)
# print('prediction:', prediction)
# model_performance = spacy_model.test(with_testing_csv='manual_training_data/manual_validation_data.csv')
# spacy_model.safe_predictions_to_csv(to='model_evaluation/spacy_v5.csv', prediction_results=model_performance)

# spacy_model.train(50, 'manual_training_data/per_loc_1.csv')

# trained_spacy_model = SpacyModel('models/spacy_0')
# trained_spacy_model = SpacyModel('de_core_news_sm')
# model = trained_spacy_model.get_model()
# f = load_data('manual_training_data/manual_validation_data.csv')[3]
# print('f:', f)
# prediction = trained_spacy_model.predict(model, f)
# print('prediction:', prediction)


# model_performance = trained_spacy_model.test(with_testing_csv='manual_training_data/per_loc_validation_1.csv')
# trained_spacy_model.safe_predictions_to_csv(to='model_evaluation/spacy.csv', prediction_results=model_performance)


# predictions = test_model(path_to_test_data='manual_training_data/manual_validation_data.csv', model='models/ner_training_with_163_data')
# write_model_test_to_csv(predictions, output_path='model_evaluation/ner_.csv')

# test_flair_model()
