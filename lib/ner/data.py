import csv
import random

from lib.ner.architecture import Fragment, EntityLabel, Entity


def load_data(from_csv: str, skip_overlapping: bool = False) -> list[Fragment]:
    fragments: list[Fragment] = []

    with open(from_csv, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        labels = get_entities(header[1:])

        for row in reader:
            entities: list[Entity] = []
            title = row[0]
            raw_entities = row[1:]
            for index, raw_entity in enumerate(raw_entities):
                try:
                    entity = Entity(text=raw_entity.strip(), label=labels[index], title=title)
                    entities.append(entity)
                except:
                    continue  # skip all empty or invalid entities

            fragment = Fragment(text=title, entities=entities)
            if skip_overlapping and fragment.entities_overlap():
                print('skipping:', fragment)
                continue

            fragments.append(fragment)

    return fragments


def get_entities(entity_labels: list[str]) -> list[EntityLabel]:
    labels: list[EntityLabel] = []
    for label in entity_labels:
        if label in EntityLabel.__members__:
            labels.append(EntityLabel[label])
        else:
            raise LookupError(f'csv file has invalid entity in header: "{label}". '
                              f'Add it to the EntityLabel enum or change the csv file.')
    return labels


### NER TRAINING DATA GENERATION


def get_list_from_csv(path: str, column_name: str, delimiter: str = ',') -> list:
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        header = next(reader)
        index = header.index(column_name)
        buffer = [row[index] for row in reader]
        random.shuffle(buffer)
        return buffer


def client_generator():
    buffer = []
    while True:
        if len(buffer) > 0:
            clt = buffer.pop()
            if clt:
                yield clt, clt
            continue
        buffer = get_list_from_csv('../../data/ner/manual_data.csv', 'CLT')


def loc_generator():
    buffer = []
    while True:
        if len(buffer) > 0:
            loc = buffer.pop()
            if loc:
                yield loc, loc
            continue
        buffer = get_list_from_csv('../../data/ner/manual_data.csv', 'LOC')


def mst_generator():
    buffer = []
    templates = [
        'Masstab $MST',
        'Masstab: $MST',
        'Mst: $MST',
        'M. $MST',
        'M: $MST',
        'M. = $MST',
        'M = $MST',
    ]
    while True:
        if len(buffer) > 0:
            mst = buffer.pop()
            if mst:
                yield random.choice(templates).replace('$MST', mst), mst
            continue
        buffer = get_list_from_csv('../../data/ner/training_data.csv', 'MST', delimiter='\t')


def cloc_generator():
    buffer = []
    while True:
        if len(buffer) > 0:
            loc = buffer.pop()
            if loc:
                yield loc, loc
            continue
        buffer = get_list_from_csv('../../data/ner/training_data.csv', 'CLOC', delimiter='\t')



def date_generator():
    buffer = []
    while True:
        if len(buffer) > 0:
            date = buffer.pop()
            if date:
                yield date, date
            continue
        buffer = get_list_from_csv('../../data/ner/training_data.csv', 'DATE', delimiter='\t')


def template_generator():
    templates = (
        [{'template': 'Garten $CLT', 'tokens': ['$CLT']}] * 9 +
        [{'template': 'Garten $CLT in $LOC', 'tokens': ['$CLT', '$LOC']}] * 6 +
        [{'template': 'Garten $CLT, $MST', 'tokens': ['$CLT', '$MST']}] * 6 +
        [{'template': 'Garten $CLT in $LOC, $MST', 'tokens': ['$CLT', '$LOC', '$MST']}] * 6 +
        [{'template': '$CLT', 'tokens': ['$CLT']}] * 2 +
        [{'template': '$LOC', 'tokens': ['$LOC']}] * 2 +
        [{'template': '$CLOC, $DATE', 'tokens': ['$CLOC', '$DATE']}] * 5 +
        [{'template': '$DATE $CLOC', 'tokens': ['$DATE', '$CLOC']}] * 5 +
        [{'template': '$CLOC am $DATE', 'tokens': ['$CLOC', '$DATE']}] * 4 +
        [{'template': '$CLOC', 'tokens': ['$CLOC']}] * 7 +
        [{'template': '$DATE', 'tokens': ['$DATE']}] * 7
    )
    while True:
        yield random.choice(templates)


def generate_ner_training_data(path: str, amount: int):
    template_gen = template_generator()
    client_gen = client_generator()
    loc_gen = loc_generator()
    mst_gen = mst_generator()
    cloc_gen = cloc_generator()
    date_gen = date_generator()

    with open(path, 'x') as file:
        writer = csv.writer(file, delimiter='\t')
        header = ['sentence', 'CLT', 'LOC', 'MST', 'CLOC', 'DATE']
        writer.writerow(header)

        for i in range(amount):
            template = next(template_gen)
            template_string = template['template']
            template_tokens = template['tokens']

            generators = {
                '$CLT': client_gen,
                '$LOC': loc_gen,
                '$MST': mst_gen,
                '$CLOC': cloc_gen,
                '$DATE': date_gen,
            }

            row = [''] * 6

            for token in template_tokens:
                replacement_str, replacement_token = next(generators[token])
                token_index = header.index(token[1:])
                row[token_index] = replacement_token
                template_string = template_string.replace(token, replacement_str)

            row[0] = template_string
            writer.writerow(row)


# generate_ner_training_data('../../data/ner/training_set_250.csv', 250)
