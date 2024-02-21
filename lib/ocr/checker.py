from spellchecker import SpellChecker


class Checker:
    def __init__(self, whitelist: list[str]):
        self.checker = SpellChecker(language='de')
        self.checker.word_frequency.load_text_file('german_dict.txt')
        # self.checker.word_frequency.load_text_file('lib/ocr/german_dict.txt')
        # self.checker.word_frequency.load_words(whitelist)

    def check_sentence(self, sentence: str) -> str:
        misspelled = self.checker.unknown(sentence.split(' '))
        print('misspelled', misspelled)
        corrected = []

        for word in misspelled:
            # Get the one `most likely` answer
            # prob = self.checker.word_probability(word)
            # print(prob)
            alternative = self.checker.correction(word)
            print(f'"{word}" should be: {alternative}')

        return ' '.join(corrected)

    def add_to_whitelist(self, new_words: list[str]) -> None:
        self.checker.word_frequency.load_words(new_words)


checker = Checker(whitelist=[])
print(checker.check_sentence('plan no : 3402 zuric 1. Dezember 1944'))
