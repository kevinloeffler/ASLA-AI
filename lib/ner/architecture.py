import csv
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class EntityLabel(Enum):
    # OLD ENTITIES:
    CLT = 'Client',
    LOC = 'Location',
    MST = 'Scale',
    CLOC = 'Creation-place'
    DATE = 'Date'

    # PER = 'Person'
    # LOC = 'Location'
    # MISC = 'Miscellaneous'
    # ORG = 'Organisation'


@dataclass
class Entity:
    """
    Holds all the data about a specific token or entity.

    text : str
        a string that contains the token, eg: 'Walter Leder'
    label: EntityLabel
        an enum that describes what kind of entity this is, eg: LOC (location)
    start_index: int
        where this entity starts in the string
    end_index: int
        where this entity ends in the string (points to the first blank char after the entity)
    _____

    raise : ValueError
        when text is empty
    raise : IndexError
        when the text can't be found in the title

    """
    text: str
    label: EntityLabel
    start_index: int
    end_index: int

    def __init__(self, text: str, label: EntityLabel, title: str):
        self.text = text
        self.label = label
        self.start_index = self.__find_start_index(text, title)
        self.end_index = self.start_index + len(text)

    @staticmethod
    def __find_start_index(text, title):
        if len(text) == 0:
            raise ValueError(f'entitiy is empty, title: "{title}"')

        start_index = title.find(text)
        if start_index == -1:
            raise IndexError(f'Start index of text "{text}" could not be found in title "{title}"')
        return start_index

    def __repr__(self):
        return f'Entity("{self.text}", {self.label.name}, ({self.start_index},{self.end_index}))'


@dataclass
class Fragment:
    text: str
    entities: list[Entity]

    def entities_overlap(self) -> bool:
        local_entities = [entity for entity in self.entities]
        entity_to_check = None
        while len(local_entities) > 0:
            entity_to_check = local_entities.pop(0)
            for entity in local_entities:
                if (entity.start_index < entity_to_check.start_index < entity.end_index
                        or entity_to_check.start_index < entity.start_index < entity_to_check.end_index
                        or entity.start_index < entity_to_check.end_index < entity.end_index
                        or entity_to_check.start_index < entity.end_index < entity_to_check.end_index):
                    return True
        return False


@dataclass
class PredictionResult:
    accuracy: float | None
    precision: float | None
    recall: float | None

    entity_accuracy: dict[str, float]

    text: str
    entities: list[Entity]
    predictions: list[Entity]


class AbstractModel(ABC):

    @abstractmethod
    def train(self, iterations: int, with_training_csv: str, safe_to: str) -> list:
        """Train the layout_model and return a list of losses"""
        return NotImplemented

    @abstractmethod
    def predict(self, model, fragment: Fragment) -> PredictionResult:
        """Make a prediction and return the result"""
        return NotImplemented

    @abstractmethod
    def test(self, with_testing_csv: str) -> list[PredictionResult]:
        """Test layout_model with a dataset"""
        return NotImplemented

    ########## UTIL ##########

    @staticmethod
    @abstractmethod
    def convert_fragment_to_data(fragment: Fragment) -> any:
        """Convert a fragment to the datastructure the layout_model expects"""
        return NotImplemented

    @staticmethod
    def safe_predictions_to_csv(to: str, prediction_results: list[PredictionResult]):
        return safe_predictions_to_csv(to, prediction_results)

    def evaluate_prediction(self, fragment: Fragment, predicted_entities: list[Entity]) -> PredictionResult:
        return evaluate_prediction(fragment, predicted_entities)


##########


def evaluate_prediction(fragment: Fragment, predicted_entities: list[Entity]) -> PredictionResult:
    true_positives = 0
    false_positives = 0

    for prediction in predicted_entities:
        for target in fragment.entities:
            if prediction.text in target.text and prediction.label.name == target.label.name:
                true_positives += 1
                break
        else:
            false_positives += 1
            continue

    false_negatives = len(fragment.entities) - true_positives

    prediction_result = PredictionResult(accuracy=1, precision=1, recall=1,
                                         entity_accuracy={},
                                         text=fragment.text,
                                         entities=fragment.entities,
                                         predictions=predicted_entities)

    if true_positives == 0 and (false_positives + false_negatives) == 0:
        return prediction_result

    entity_accuracy = evaluate_entity_accuracy(fragment=fragment, predicted_entities=predicted_entities)
    prediction_result.entity_accuracy = entity_accuracy

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.

    prediction_result.accuracy = f1_score
    prediction_result.precision = precision
    prediction_result.recall = recall
    return prediction_result


def evaluate_entity_accuracy(fragment: Fragment, predicted_entities: list[Entity]) -> dict[str, float]:
    relevant_entities = set(
        [entity.label.name for entity in fragment.entities] + [entity.label.name for entity in predicted_entities]
    )
    entity_accuracy = {key: 0.0 for key in relevant_entities}

    for entity in relevant_entities:
        targets = list(filter(lambda t: t.label.name == entity, fragment.entities))
        predictions = list(filter(lambda p: p.label.name == entity, predicted_entities))

        true_positives = 0
        false_positives = 0

        for prediction in predictions:
            for target in targets:
                if prediction.text in target.text:
                    true_positives += 1
                    break
            else:
                false_positives += 1
                continue

        false_negatives = len(targets) - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        entity_accuracy[entity] = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.

    return entity_accuracy


def safe_predictions_to_csv(to: str, prediction_results: list[PredictionResult]):
    with open(to, 'x') as file:
        writer = csv.writer(file)
        writer.writerow(['title', 'accuracy', 'entity_accuracy', 'predictions', 'targets'])

        for prediction in prediction_results:
            writer.writerow([prediction.text,
                             prediction.accuracy,
                             prediction.entity_accuracy,
                             prediction.predictions,
                             prediction.entities])
