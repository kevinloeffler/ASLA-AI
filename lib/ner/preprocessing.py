import csv


def extract_list_of_historic_municipalities(path_to_file: str):
    with open(path_to_file, 'r', encoding='ISO-8859-1') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        output = []
        for row in reader:
            output.append(row[4])
        random_output = []
        for i in range(100):
            random_output.append(output)
        print('\n'.join(output[300: 500]))
        return output[300: 500]


def print_municipalities(path_to_examples: str):
    locations = extract_list_of_historic_municipalities(
        '../../data/ner/manual_training_data/generated_historic/gemeinden_ch/gemeinden_1960/20230101_GDEHist_GDE.tsv')
    examples = []
    with open(path_to_examples, 'r') as file:
        for row in file.readlines():
            examples.append(row.replace('\n', ''))

    for i in range(len(examples)):
        print(examples[i] + '||' + locations[i])

    print(len(examples))


# print_municipalities('../manual_training_data/chat_gpt_location_files/200-300.txt')
# extract_list_of_historic_municipalities('../manual_training_data/gemeinden_ch/gemeinden_1960/20230101_GDEHist_GDE.tsv')
