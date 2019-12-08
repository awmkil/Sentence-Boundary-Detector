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
        self.begin_quote_punctuation = '''([{'"<'''
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
        - left_right_couples: Words before and after an "end_of_sentence_punctuation" character.
        - labels: Label of each "left_right_couple" Can either be MOS (middle of sentece) or EOS (end of sentence).
        - sentences_index: Sentece index.
        """

        left_right_couples = []
        labels = []
        sentences_index = []

        # Loop on the dataset's sentences
        for sentence_index, sentence in tqdm(enumerate(self.data_set)):

            # Word tokenise each sentence
            sentence = get_word_list_tokenized(sentence)

            # Loop on each sentence's words
            for word_index, word in enumerate(sentence):

                # Look for end of sentence punctuation
                if re.search(self.end_of_sentence_punctuation, word):

                    # Save sentence index
                    sentences_index.append(sentence_index)

                    # Case where an end punctuation is the last character of a sentence (EOS case)
                    if word_index == len(sentence) - 1:
                        # Label EOS
                        labels.append("EOS")

                        # Edge case when we reach the last sentence.
                        if sentence_index == len(self.data_set) - 1:
                            left_right_couples.append(
                                (sentence[word_index - 1], ""))

                        # Save the left and right word couple.
                        else:
                            left_right_couples.append(
                                (
                                    sentence[word_index - 1],
                                    self.data_set[sentence_index + 1][0],
                                )
                            )

                    # Case where an end punctuation is not the last character of a sentence (MOS case)
                    else:
                        # Label MOS
                        left_right_couples.append(
                            (sentence[word_index - 1],
                             sentence[word_index + 1])
                        )
                        labels.append("MOS")

        return left_right_couples, labels, sentences_index

    def get_features(self, left_right_couple):
        """
        Extracts relevant features from each left_right_couple.
        These will feed the classifier.
        """

        return {
            "left_word": left_right_couple[0],
            "right_word": left_right_couple[1],
            "left_word_length": len(left_right_couple[0]),
            # "right_word_lenght": len(left_right_couple[1]),
            "is_right_word_upper": left_right_couple[1][:1].isupper(),
            # "logn_right_word_capital_occurences": int(np.log(self.data_set.count(word_pair[1]))),
        }

    def train_model(self):
        """
        Creates the featureset and trains the model using NLTK's NaiveBayesClassifier.
        """

        # Get training data
        print("Getting training data...")
        left_right_couples, labels, sentence_number = self.get_training_data()

        # Get and format featureset
        print("Extracting features...")
        featuresets = [
            (self.get_features(left_right_couple), label)
            for (left_right_couple, label) in zip(left_right_couples, labels)
        ]

        # Divide into train and test set
        train_size = 0.8
        train_limit_idx = int(len(featuresets)*train_size)
        train_set, test_set = featuresets[:train_limit_idx], featuresets[train_limit_idx:]
        # Train classifier and print accuracy
        print("Training...")
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
        print("Success! Accuracy: " +
              str(nltk.classify.accuracy(self.classifier, test_set)))

    def save_model(self, model_path):
        """
        Writes the pickled representation of the model to a file.
        """
        print("Saving model to : " + model_path)
        with open(model_path, "wb") as f:
            pickle.dump(self.classifier, f)

    def load_model(self, model_path):
        """
        Loads the pickled representation of the model from a file.
        """
        print("Loading model from : " + model_path)
        with open(model_path, "rb") as f:
            self.classifier = pickle.load(f)

    def read_html(self, htlm_file_path):
        """
        Given a HTML file, converts it into a BeautifulSoup object.
        """
        print("Reading HTML from : " + htlm_file_path)
        with open(htlm_file_path, "r") as source:
            html = BeautifulSoup(source, features="html.parser")
            return html

    def get_candidates(self, text):
        """
        Gets candidates to be labeled from a text, and their positions.
        """

        left_right_couples_candidates = []
        positions = []

        # Word tokenize
        text = get_word_list_tokenized(nltk.word_tokenize(text))

        # Loop on the words
        for word_index, word in enumerate(text):

            # Look for end of sentence punctuation
            if re.search(self.end_of_sentence_punctuation, word):

                # Save candidate position
                positions.append(word_index)

                # Edge case when we reach the last sentence.
                if word_index == len(text) - 1:
                    left_right_couples_candidates.append(
                        (text[word_index - 1], ""))

                # Save L/R couples
                else:
                    left_right_couples_candidates.append(
                        (text[word_index - 1], text[word_index + 1]))

        return left_right_couples_candidates, positions


    def label_HTML(self, htlm_file_path):
        """
        Given an HTML file, wrap each sentence in a <span> tag and returns the modified HTML file.
        """

        # Read HTML
        html_soup = self.read_html(htlm_file_path)
        print("Processing HTML file...")
        
        # Init prediction list
        predictions = []

        # Loop on the HTML text portions
        for text in html_soup.find_all(text=True):

                # Copy text portion to avoid dynamic typing problems
            str_text = str(text)

            # For each text portion get candidates and their positions
            left_right_couples_candidates, positions = self.get_candidates(
                str_text)

            # Extract features from candidates and predict labels using the classifier
            for index, left_right_couples_candidate in enumerate(left_right_couples_candidates):
                predictions.append(self.classifier.classify(
                    self.get_features(left_right_couples_candidate)))

            # Word tokenize the text portion
            input_text_words = get_word_list_tokenized(
                nltk.word_tokenize(str_text))

            # String with text + wrapper
            output_text = ""

            # Edge case for empty text portions / portions with no candidates (no EOS punctuation)
            if positions == [] or len(str_text) < 2:

                # Loop on text words
                for i in input_text_words:

                    # Copy with no space before if the word is a punctuation word
                    if i in string.punctuation:
                        output_text += "".join([i])

                    # Else, copy with a space before
                    else:
                        output_text += "".join([" " + i])

            # Nominal case, the text portion contains candidates
            else:
                # Open wrapper in the begining of the text portion
                output_text += "<span>"

                # Loop on candidates positions
                for index, position in enumerate(positions):

                        # Force EOS label on the last EOS punctuation of the text
                    if index == len(positions)-1:
                        predictions[index] == "EOS"

                    # Edge case for index 0
                    if index == 0:
                        for i in input_text_words[0: position + 1]:
                            # Copy with no space before if the word is a punctuation word.
                            if i in string.punctuation:
                                output_text += "".join([i])
                            # Else, copy with a space before
                            else:
                                output_text += "".join([" " + i])

                    # Nominal case
                    else:
                        for i, j in zip(input_text_words[positions[index - 1] + 1: position + 1], input_text_words[positions[index - 1]: position + 1]):
                            # Copy with no space before if the current word is a punctuation word or the begining of a quote.
                            if i in string.punctuation or j in self.begin_quote_punctuation:
                                output_text += "".join([i])
                            # Else, copy with a space before
                            else:
                                output_text += "".join([" " + i])

                    # Close span tag when an end of sentence is detected and open a new one.
                    if predictions[index] == "EOS":
                        output_text += "</span> <span>"

                    # Signal when a MOS punctuation character is detected.
                    else:
                        output_text += "[MOS]"

                # Delete last <span/> at the end of paragraph.
                output_text = output_text[:-6]

            # Replace text in BeautifulSoup object.
            text.replace_with(output_text)

        # Pretiffy and save to new object.
        new_html_data = html_soup.prettify()

        return new_html_data

    def save_HTML(self, new_html_data, htlm_file_path):
        """
        Writes BeautifulSoup to HTML file.
        """
        print("Segmented HTML file saved to : " + htlm_file_path)
        with open(htlm_file_path, 'w') as f:
            f.write(new_html_data)


if __name__ == "__main__":
    from argparse import ArgumentParser

    argparser = ArgumentParser(
        prog="python3 -m sentence_boundaries_detector", description="Sentence boundary detector for HTML ebooks."
    )

    argparser.add_argument(
        "-l", "--load", help="Run the program using a trained model saved to your computer.")
    argparser.add_argument(
        "-t", "--train", help="Train and write out serialized model.")
    argparser.add_argument(
        "-i", "--input", help="Segment sentences from an input HTML file.")
    argparser.add_argument(
        "-s", "--save", help="Save semented HTML file.")

    args = argparser.parse_args()

    # Input block
    detector = Detector()

    if args.load:
        detector.load_model(args.load)

    else:
        detector.train_model()
        detector.save_model(args.train)

    # Output block
    if args.input:
        new_html_data = detector.label_HTML(args.input)

    if args.save:
        detector.save_HTML(new_html_data, args.save)
