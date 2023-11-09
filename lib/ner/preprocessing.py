import csv
import os
import re


def combine_nachlaesse(output_file: str):
    entities = ['HEAD', 'MST', 'DATE']
    data = {
        'cramer-ernst': {
            'path': 'nachlaesse/csv/Werkverzeichnis-Cramer-Ernst.csv', 'id': 'identifier', 'project': 'cre',
            'labels': {'HEAD': 'plan_head', 'MST': 'note.scale', 'DATE': 'creation_date_start'}
        },
        'mertens-nussbaumer': {
            'path': 'nachlaesse/csv/Werkverzeichnis-Mertens-Nussbaumer.csv', 'id': 'identifier', 'project': 'mnu',
            'labels': {'HEAD': 'plan_head', 'MST': 'note.scale', 'DATE': 'creation_date_start'}
        },
        'klauser': {
            'path': 'nachlaesse/csv/Werkverzeichnis-Klauser.csv', 'id': 'identifier', 'project': 'kla',
            'labels': {'HEAD': 'plan_head', 'MST': 'scale', 'DATE': 'creation_date_start'}
        },
    }

    with open(output_file, 'x') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['project', 'id'] + entities)

        for architect, info in data.items():
            with open(info['path'], 'r') as file:
                reader = csv.reader(file)
                header = next(reader)
                id_index = header.index(info['id'])
                head_index = header.index(info['labels']['HEAD'])
                mst_index = header.index(info['labels']['MST'])
                date_index = header.index(info['labels']['DATE'])
                for row in reader:
                    writer.writerow([
                        info['project'],
                        row[id_index],
                        row[head_index],
                        format_mst(row[mst_index]),
                        row[date_index],
                    ])


def format_mst(mst: str) -> str | None:
    # Condition 1 - correct pattern: 1:100
    if re.match(r'^\d+:\d+$', mst):
        return mst

    # Condition 2 - leading '1:': 1:1:100
    if re.match(r'^\d+:\d+:\d+$', mst):
        return re.sub(r'^\d+:', '', mst)

    # Condition 3 - number only: 100
    if re.match(r'^\d+$', mst):
        return f'1:{mst}'

    return None



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
