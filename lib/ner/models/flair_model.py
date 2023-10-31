from lib.ner.architecture import AbstractModel, PredictionResult, Fragment


class FlairModel(AbstractModel):

    model_name: str

    def __init__(self, model_name: str):
        self.model_name = model_name

    ########## TRAINING ##########

    def train(self, iterations: int, with_training_csv: str, safe_to: str) -> list:
        pass

    ########## TESTING ##########

    def predict(self, model, fragment: Fragment) -> PredictionResult:
        pass

    def test(self, with_testing_csv: str) -> list[PredictionResult]:
        pass

    ########## UTIL ##########

    def __create_data_from_fragment(self, fragment: Fragment) -> tuple[str, str | None, str | None]:
        test = ('plantitel', None, 'Rapperswil')
        locations = list(filter(lambda e: e.label))

        return fragment.text, None, None





