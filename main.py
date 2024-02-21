import tomli


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

    # train_spacy(config=config)
    # evaluate_spacy(config=config)
    # train_transformer_model(config=config)
    # evaluate_transformer_model(config=config)

    #'''
    from lib.ner.architecture import Fragment
    from lib.ner.models.transformer_model import TransformerModel

    model = TransformerModel(model_type='bert',
                             model_name=config['paths']['models'] + 'ner/german-bert-500',
                             numbers_of_gpus=config['number_of_gpus'],
                             training_iterations=1,
                             gpu_id=config['gpu_id'],
                             safe_to='')
    model.predict(Fragment(text='Thuyahecke ) dahlier Herbst. stauden. kompost. s. abdeckery. zurich, 10. Nov. 1945. samling. plan. -NR. 3568', entities=[]))
    #'''
    '''
    from lib.ner.architecture import Fragment
    from lib.ner.models.spacy_model import SpacyModel

    spacy = SpacyModel(model_name=config['paths']['models'] + 'ner/spacy_4')
    model = spacy.get_model()
    spacy.predict(model, Fragment(text='archiv. fur. die. schweizer. Gartenarchitektur und. landschaftsplanung Rapperswil. sg. Nachlass. ernst. cramer. Norveys own history', entities=[]))
    '''
    '''
    from lib.ner.architecture import Fragment
    from lib.ner.models.transformer_model import TransformerModel

    model = TransformerModel(model_type='bert',
                             model_name=config['paths']['models'] + 'ner/german-bert-500',
                             numbers_of_gpus=config['number_of_gpus'],
                             training_iterations=1,
                             safe_to='',
                             gpu_id=config['gpu_id'])
    model.predict(Fragment(text='larus. zurigh in- it- r. 2. 8 2. der. gartenarch.', entities=[]))
    model.predict(Fragment(text='drell. istraire.', entities=[]))
    model.predict(Fragment(text='0. e. Horizontalen.', entities=[]))
    model.predict(Fragment(text='das. landhausgebiet. gesamplan 1,500 stand. Der-bebauung-Herbst 1928.', entities=[]))
    model.predict(Fragment(text='susenbergstrasse masstab. 1 : 500 40 50. 60. 90. 100 m.', entities=[]))
    model.predict(Fragment(text='gruman', entities=[]))
    '''

    # OCR
    # model = OCR(layout_model='microsoft/layoutlmv2-base-uncased', ocr_model='microsoft/trocr-large-handwritten')
    # model.predict('data/ocr/cre/cre_6202_1.jpg', overlap_threshold=0.5)

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
