import tomli

from lib.ocr.models.OCR import OCR
from scripts.train_spacy import train_spacy

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

    train_spacy(config)

    # OCR
    # model = OCR(layout_model='microsoft/layoutlmv2-base-uncased', ocr_model='microsoft/trocr-large-handwritten')
    # model.predict('data/ocr/mnu/MN_1517_3.jpg', overlap_threshold=0.5)

    # data = load_images(csv_file=config['paths']['data'] + 'ner/training_data.csv',
                       #image_base_path=config['paths']['data'] + 'ocr/', count=4)
    # image, metadata = data[2]

    # image = Image.open('data/ocr/ocr_test_img.jpg')

    # result_image = run_inference('data/ocr/kla/KLA_6533_00001.JPG')
    # result_image.show()

    # layout_model = TrOCR_2(config=config, layout_model='microsoft/trocr-base-handwritten')
    # layout_model.predict(image)

    # data = load_images(csv_file=config['paths']['data'] + 'ner/training_data.csv',
    #                   image_base_path=config['paths']['data'] + 'ocr/')

    # evaluate_transformer_model(config)
    # train_transformer_model(config)


# data = layout_model.load_data('data/ner/manual_training_data/per_loc_1.csv')
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
# layout_model = trained_spacy_model.get_model()
# f = load_data('manual_training_data/manual_validation_data.csv')[3]
# print('f:', f)
# prediction = trained_spacy_model.predict(layout_model, f)
# print('prediction:', prediction)


# model_performance = trained_spacy_model.test(with_testing_csv='manual_training_data/per_loc_validation_1.csv')
# trained_spacy_model.safe_predictions_to_csv(to='model_evaluation/spacy.csv', prediction_results=model_performance)


# predictions = test_model(path_to_test_data='manual_training_data/manual_validation_data.csv', layout_model='models/ner_training_with_163_data')
# write_model_test_to_csv(predictions, output_path='model_evaluation/ner_.csv')

# test_flair_model()
