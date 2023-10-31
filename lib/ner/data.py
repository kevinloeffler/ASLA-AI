import csv

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
