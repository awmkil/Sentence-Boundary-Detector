#!/usr/bin/env python3

import re
import nltk
import pickle
import string
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_word_list_tokenized(word_list):
    word_list_tokenized = []
    for word in word_list:
        word_list_tokenized.extend(nltk.word_tokenize(word))
    return word_list_tokenized


class Detector:
    def __init__(self):
        self.end_of_sentence_punctuation = r"([.!?]+)$"
        self.quotes_punctuation= '''([{'"<'''
        self.data_set = []
        self.classifier = None

        self.get_data_set()

    def get_data_set(self):
        """
        Downloads the following ressources and formats the dataset:
        - Brown Corpus: General text collection. 500 samples of English text (~1M words).
                    Used for training and testing.

        - Punkt: Punkt sentence tokenization models used by "nltk.word_tokenize"
        """

        try:
            nltk.data.find('corpora/brown')
        except LookupError:
            nltk.download('brown')

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # Format the dataset into a list of sentences
        self.data_set = nltk.corpus.brown.sents()

    def get_training_data(self):
        """
        Extracts the following data from the dataset:
        - left_right_couples: Words before and after an end_of_sentence_punctuation character.
        - labels: Label of each "left_right_couple" Can either be MOS (middle of sentece) or EOS (end of sentence).
        - sentence_number: The sentence number
        """

        left_right_couples = []
        labels = []
        sentence_number = []

        # Loop on the dataset's sentences
        for sentence_index, sentence in tqdm(enumerate(self.data_set)):
            # Word tokenise
            sentence = get_word_list_tokenized(sentence)
            # Loop on each sentence's words
            for word_index, word in enumerate(sentence):
                # Look for end of sentence punctuation
                if re.search(self.end_of_sentence_punctuation, word):
                    # Save sentence number
                    sentence_number.append(sentence_index)
                    # Case where end punctuation is an EOS
                    if word_index == len(sentence) - 1:
                        # Label EOS
                        labels.append("EOS")
                        # Edge case when we reach the last sentence. Managed by making the right word ""
                        if sentence_index == len(self.data_set) - 1:
                            left_right_couples.append(
                                (sentence[word_index - 1], ""))
                        # Save the left and right words.
                        else:
                            left_right_couples.append(
                                (
                                    sentence[word_index - 1],
                                    self.data_set[sentence_index + 1][0],
                                )
                            )
                    # Case where end punctuation is an EOS
                    else:
                        # Label is MOS otherwise
                        left_right_couples.append(
                            (sentence[word_index - 1],
                             sentence[word_index + 1])
                        )
                        labels.append("MOS")

        return left_right_couples, labels, sentence_number

    def get_features(self, left_right_couple):
        """
        Extracts relevant features from each left_right_couple.
        """
        return {
            "left_word": left_right_couple[0],
            "right_word": left_right_couple[1],
            "left_word_length": len(left_right_couple[0]),
            "right_word_lenght": len(left_right_couple[1]),
            "is_right_word_upper": left_right_couple[1][:1].isupper(),
            # "logn_right_word_capital_occurences": int(np.log(self.data_set.count(word_pair[1]))),
        }

    def train_model(self):
        """
        Creates the featureset and trains the model using NLTK's NaiveBayesClassifier.
        """
        left_right_couples, labels, sentence_number = self.get_training_data()

        # Get and format featureset
        for (left_right_couple, label) in zip(left_right_couples, labels):

            featuresets = [(self.get_features(left_right_couple), label)]

        # Divide into training and test set, and training classifier
        train_size = 0.8
        train_limit_idx = int(len(featuresets)*train_size)
        train_set, test_set = featuresets[:train_limit_idx], featuresets[train_limit_idx:]

        # Train classifier and print accuracy
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
        print("Accuracy: " + str(nltk.classify.accuracy(self.classifier, test_set)))

    def save_model(self, model_path):
        """
                Writes the pickled representation of the model to a file
        """
        print("Saving model to : " + model_path)
        with open(model_path, "wb") as f:
            pickle.dump(self.classifier, f)

    def load_model(self, model_path):
        """
                Loads the pickled representation of the model from a file
        """
        print("Loading model from : " + model_path)
        with open(model_path, "rb") as f:
            self.classifier = pickle.load(f)

    def read_html(self, htlm_file_path):
        """
        Given a `filename` string, convert the whole file into a string
        """
        with open(htlm_file_path, "r") as source:
            html = BeautifulSoup(source, features="html.parser")
            return html

    def get_candidates(self, text):
        """
        Gets candidates to be labeled from a text, and their positions.
        """
        left_right_couples_candidates = []
        positions = []

        text = get_word_list_tokenized(nltk.word_tokenize(text))

        for word_index, word in enumerate(text):
            # Look for end of sentence punctuation
            if re.search(self.end_of_sentence_punctuation, word):
                # Save candidate position
                positions.append(word_index)
                # Edge case when we reach the last sentence. Managed by making the right word ""
                if word_index == len(text) - 1:
                    left_right_couples_candidates.append(
                        (text[word_index - 1], ""))
                # Nominal case
                else:
                    left_right_couples_candidates.append(
                        (text[word_index - 1], text[word_index + 1]))

        return left_right_couples_candidates, positions

    def label_obvious_candidates(self, left_right_couples_candidates):
        """
        This function labels the couples where the right word is "<" as EOS
        since when a balise is present after a sentence, it means that the
        paragraph is over.
        """

    def label_candidates_for_text(self, htlm_file_path):
        html = self.read_html(htlm_file_path)
        text = html.get_text('\n')

        left_right_couples_candidates, positions = self.get_candidates(text)

        predictions = []
        span_tag = html.new_tag('span')

        # Get the features and classify
        for index, left_right_couples_candidate in enumerate(left_right_couples_candidates):
            predictions.append(self.classifier.classify(
                self.get_features(left_right_couples_candidate)))

        input_text_words = get_word_list_tokenized(nltk.word_tokenize(text))
        output_text = ""

        for index, position in enumerate(positions):
            if index == 0:
            	for i in input_text_words[0: position + 1]:
                	# Ne pas insérer d'éspaces avant une ponctuation ou avant le mot suivant un [(<
                    if i in string.punctuation:
                        output_text += "".join([i])
                    else:
                        output_text += "".join([" " + i])

            else:
                for i, j in zip(input_text_words[positions[index - 1] + 1: position + 1], input_text_words[positions[index - 1]: position + 1]):
                	# Ne pas insérer d'éspaces avant une ponctuation ou avant le mot suivant un [(<
                    if i in string.punctuation or j in self.quotes_punctuation:
                        output_text += "".join([i])
                    else:
                        output_text += "".join([" " + i])

            if predictions[index] == "EOS":
                output_text += "<EOS>\n\n"

            else:
                output_text += "<MOS>"

        return output_text


detector = Detector()
# detector.train_model()
# detector.save_model("./model.pickle")
detector.load_model("./model.pickle")

output = detector.label_candidates_for_text("./test_book.html")

with open("output.txt", "w") as f:
    f.write(output)

# print(left_right_couples_candidates)