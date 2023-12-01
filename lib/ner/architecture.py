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
        an enum that describes what kind of entity this is, eg: PER (person), LOC (location)
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
    prediction_result = PredictionResult(accuracy=None,
                                         entity_accuracy={},
                                         text=fragment.text,
                                         entities=fragment.entities,
                                         predictions=predicted_entities)
    accuracies = []

    labels = set([p.label.name for p in predicted_entities] + [e.label.name for e in fragment.entities])
    for label in labels:
        targets = list(filter(lambda e: e.label.name == label, fragment.entities))
        predictions = list(filter(lambda p: p.label.name == label, predicted_entities))
        accuracy = compare_prediction_to_target(target=targets, prediction=predictions)
        if accuracy is not None:
            accuracies.append(accuracy)
            prediction_result.entity_accuracy[label] = accuracy

    if not len(accuracies) == 0:
        prediction_result.accuracy = sum(accuracies) / len(accuracies)

    return prediction_result


def compare_prediction_to_target(target: list[Entity], prediction: list[Entity]) -> float | None:
    delimiters = '\s|,|:|;|/'

    tokenized_target = list(filter(
        lambda el: el != '', re.split(delimiters, ' '.join([t.text for t in target]))))

    tokenized_prediction = list(filter(
        lambda el: el != '', re.split(delimiters, ' '.join([p.text for p in prediction]))))

    total_tokens = len(tokenized_target)
    correct_tokens = 0
    incorrect_tokens = 0

    for t_target in tokenized_target:
        if t_target in tokenized_prediction:
            correct_tokens += 1
            tokenized_prediction.remove(t_target)
        else:
            incorrect_tokens += 1

    incorrect_tokens += len(tokenized_prediction)

    if total_tokens == 0 and incorrect_tokens == 0:
        return None  # nothing to predict and not too much predicted
    elif total_tokens == 0:
        return 0.  # too much predicted
    else:
        return max(0., (correct_tokens - incorrect_tokens) / total_tokens)


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
