import tomli

from scripts.train_spacy import train_spacy
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

    train_spacy(config=config)
    # train_transformer_model(config=config)

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

    model = TransformerModel(model_type='roberta',
                             model_name=config['paths']['models'] + 'ner/trf_roberta_4',
                             numbers_of_gpus=config['number_of_gpus'],
                             training_iterations=1,
                             gpu_id=config['gpu_id'])
    model.predict(Fragment(text='klosterhof. wettingen. hofeinblick no. # 20 1949', entities=[]))
    model.predict(Fragment(text='archiv. fur. bile. schweizer. Gartenarchitektur und. landschaftsplanung rapperswil. sq. sammlung. mertens,n. " September', entities=[]))
    model.predict(Fragment(text='mertens. thussbauer Gartchurch. bsg. swb.', entities=[]))
    model.predict(Fragment(text='klosterhof. wettingen. m. 1; 100, projekt. zur. umgestaltung des. hofes.', entities=[]))
    model.predict(Fragment(text='zurich 16. 2.49. plan. no, #419 "', entities=[]))
    model.predict(Fragment(text='mertens # Nussbaumef gartenarch. b"s.g. f', entities=[]))
    model.predict(Fragment(text='archiv. fur. die. schweizer. gartenarchitektur. und. landschaftsplanung rapperswil. sg. sammlung. mertens Nussbaumer', entities=[]))
    model.predict(Fragment(text='1940s 1934 1907 1939 References THEA E. the first of a number of # Eisenhaq # # umgestell 0 0 1940s. 36.30 30. 0. 0 0 1907 1903 s. this Deckstraucher 0. 1961. to irri Brutenstand. 0 : ri- sci. # # 95 0 0 1907 08 57 " 53 " 4tho. \' 354-97. 1907 08 1907 1910 sion. 5 1 transiting 0 States 1907 0 0 Obstspaliere. ofavor. k. kirchenkrauter it 0 0 The 0 0 Prockenmal auer. s # # 0 0 0 0 0 0 0 Gebrider Mertens p 0 0 Gartenarchitekten B.S.G. Seyman\'s 1934 0 0 It is', entities=[]))
    model.predict(Fragment(text='Garten zum. Pfarrhaus Herrliberg. Norden 0 0 0 4tho Wiese mit. Obstbaumen archiv. IDECHAFTSPIL Rappers Wil SG... sameung Martens, Nussbaumer', entities=[]))
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
