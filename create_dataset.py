import os
from typing import Tuple, List
import pandas as pd
import csv
import numpy as np
from preprocessing import Sentence

# pipeline: annotate (=assign label to sentences), save to csv, undersample majority class, preprocess data, data split

class AnnotatedTexts():
    def __init__(self,
                 #step_corpus_directory: str = "/Users/laurastahlhut/Documents/Jobs/HA_DiLiUn/momo/PH_Korpus/"
                                              #"1_steps_tagging/data_analysis/I1_neu/steps_corpora",
                 step_corpus_directory: str = "/Users/laurastahlhut/Documents/Jobs/HA_DiLiUn/momo/PH_Korpus/"
                                              "1_steps_tagging/data_analysis/I1_neu/corrected",
                 texts_directory: str = "/Users/laurastahlhut/Documents/Jobs/HA_DiLiUn/momo/PH_Korpus/"
                                        "1_steps_tagging/4_tagged/I1/source",
                 ):
        self.texts_directory = texts_directory
        self.step_corpus_directory = step_corpus_directory
        self.data_directory = "data"
        self.iteration: str = "I1"  # TODO adjust when we have new data
        self.input_texts = self.read_texts()
        self.steps_corpora = self.read_steps_corpora()
        self.annotated_texts = self.annotate_texts()  # list of sentences with labels per text
        self.undersampled_data = self.undersample_data()  # df: annotated with label, undersampled class 0
        self.preprocessed = self.preprocess_annotated_texts()  # df

    def read_texts(self) -> List[Tuple[str, List[str]]]:
        """Return a list of tuples (filename, list of sentences)"""
        fnames = [f for f in os.listdir(self.texts_directory) if os.path.isfile(os.path.join(self.texts_directory, f))]
        files = []
        for file in fnames:
            with open(os.path.join(self.texts_directory, file), 'r') as infile:
                lines = infile.readlines()
                lines = [l.rstrip('\n') for l in lines]
                lines = [l.strip('"') for l in lines]
                lines = [l.strip('“') for l in lines]
                lines = [l.strip('„') for l in lines]
                files.append((file, lines))

        return files

    def read_steps_corpora(self) -> dict[int, List[str]]:
        """Returns a dictionary with steps as keys and list of tagged sentences as values.
        There are 11 classes with class 0 = not annotated."""
        steps_dictionary = {
            '1c': 'step1c_korr.txt',  # TODO change if we get another version of these files
            '1d': 'step1d_korr.txt',
            '1e': 'step1e_korr.txt',
            '2a': 'step2a_korr.txt',
            '2c': 'step2c_korr.txt',
            '3a': 'step3a_korr.txt',
            '3b': 'step3b_korr.txt',
            '3d': 'step3d_korr.txt',
            '3e': 'step3e_korr.txt',
            '3g': 'step3g_korr.txt'
        }
        annotated_sentences = {}

        for key in steps_dictionary:
            with open(os.path.join(self.step_corpus_directory, steps_dictionary[key]), 'r') as f1:
                lines = f1.readlines()
                lines = [l.rstrip('\n') for l in lines if l.strip()]
                lines = [l.strip('“') for l in lines]
                lines = [l.strip('„') for l in lines]
                annotated_sentences[key] = lines

        return annotated_sentences

    def annotate_texts(self) -> List[Tuple[str, List[Tuple[str, int]]]]:
        """Return a list of Tuples (filename, annotated sentences) where annotated sentences are a list of tuples with
        (sentence, label)"""
        annotated_texts = []
        #text_id = 1  # change filenames to integers

        for text in self.input_texts:
            filename, sentences = text
            annotated_sentences = [(s, 0) for s in sentences]  # initialize with label '0' for each sentence
            # replace label 0 with correct label if sentence is found in steps corpus
            for s in sentences:
                for tag, tagged_sents in self.steps_corpora.items():
                    if s in tagged_sents:
                        annotated_sentences[sentences.index(s)] = (s, tag)
                        break
            annotated_texts.append((text[0], annotated_sentences))  # todo replace text name with id perhaps
            #text_id += 1

        return annotated_texts

    def write_annotated_text_to_df(self):
        """Create a df with text_id, sentence, label which can be written to a csv."""
        # wite to pd dataframe
        df = pd.DataFrame(self.annotated_texts, columns=["text_id", "sentences"])
        df = df.explode("sentences")
        sent = df["sentences"].tolist()
        df[["sentence", "label"]] = pd.DataFrame(df["sentences"].tolist(), index=df.index)
        df = df.drop("sentences", axis=1)

        return df

    def undersample_data(self):
        """Undersample class labeled with 0 (sentences that aren't annotated).
            TODO: Perhaps I need to half classes 0 and 5 to make all classes more even...
            """
        df = pd.read_csv('data/data_I1.csv')
        #df = self.write_annotated_text_to_df()  # text_id, sentence, label # todo why doesnt it work with this line
        # Get a list of indeces of the majority class (0)
        class_counts = df['label'].value_counts()  # Count the number of instances in the majority class
        majority_class = class_counts.index[0]
        majority_class_count = class_counts[majority_class]
        majority_class_indices = df[df['label'] == majority_class].index.tolist()  # indeces majority class
        # # Randomly select a subset of indices from the majority class, merge with indeces of minority class
        random_indices = np.random.choice(majority_class_indices, size=500, replace=False).tolist()  # TODO check size
        minority_class_indices = df[df['label'] != majority_class].index.tolist()
        under_sampled_indices = minority_class_indices + random_indices
        # # Get the under-sampled dataframe
        under_sampled_df = df.loc[under_sampled_indices]

        return under_sampled_df

    def preprocess_annotated_texts(self):
        """Preprocess sentences (lemmatized, removed stop words and punctuation)"""
        df = self.undersampled_data
        df["sentence"] = df['sentence'].apply(lambda x: Sentence(str(x)).preprocessed)

        return df





def train_test_split(df, train_path, val_path, test_path):
    """Data split of dataframe (80/10/10 for train/test/val).
    TODO: do I need to keep all sentences of one text in the same frame (such that an entire text is either in the
    train or test dataframe, not mixed)"""
    #df = pd.read_csv(df) # uncomment in case you're reading a file (not df)
    # get random sample for test
    train = df.sample(frac=0.6, axis=0)
    # get everything but the test sample
    rest = df.drop(index=train.index)
    # split val/test 50/50
    val = rest.sample(frac=0.5, axis=0)
    test = rest.drop(index=val.index)

    # write to csv
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    val.to_csv(val_path, index=False)

    return None

def main():
    a = AnnotatedTexts()

    # 1) Write annotated data to csv
    #annotated_data = a.write_annotated_text_to_df() # df with text_id_, sentence, label
    #annotated_data.to_csv(os.path.join(a.data_directory, "data_" + a.iteration + ".csv"), index=False, sep=',')

    # 2) Write undersampled data to csv
    #a.undersampled_data.to_csv(os.path.join(a.data_directory, "data_" + a.iteration + "_undersampled.csv"),
                             #index=False, sep=',', header=True)

    # 3) Write preprocessed_data to csv
    #a.preprocessed.to_csv(os.path.join(a.data_directory, "data_" + a.iteration + "_preprocessed.csv"),
                          #index=False, sep=',', header=True)

    #5) Split preprocessed data
    train_path = 'data/train_I1.csv'
    test_path = 'data/test_I1.csv'
    val_path = 'data/val_I1.csv'
    train_test_split(a.preprocessed, train_path, val_path, test_path)




if __name__ == '__main__':
    main()
