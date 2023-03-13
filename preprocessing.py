"""Preprocess sentence
For now, i only lemmatized, removed stop words and punctuation for preprocessing.
TODO: Extract main clauses (& other preprocessing ideas)
Source
- https://subscription.packtpub.com/book/data/9781838987312/2/ch02lvl1sec13/splitting-sentences-into-clauses
- https://stackoverflow.com/questions/64055526/extract-main-and-subclauses-from-german-sentence-with-spacy
"""

import re
import spacy
from nltk.corpus import stopwords

nlp = spacy.load('de_core_news_sm')


class Sentence():
    def __init__(self, sentence):
        self.sentence = nlp(sentence)
        self.tokens = self.get_tokens()
        self.preprocessed = self.preprocess()

    def preprocess(self):
        """Remove punctuation & stopwords from string, lemmatize"""
        # lemmatize (it changes punct to '--')
        lemmas = [token.lemma_ for token in self.sentence] # TODO try out whether stemming is better than lemmatization
        # remove stopwords
        german_stop_words = stopwords.words('german')
        german_stop_words.extend(['bzw.', 'bzw']) # add bzw. to stopwords which occurs very frequently in the corpus
        # todo replace soft hyphen SHY with - or figure out where encoding error occurred
        without_sw = ' '.join([w for w in lemmas if not w  in german_stop_words])
        # remove punctuation
        without_punc = re.sub(r'\d+', '', without_sw) # remove numbers
        without_punc = re.sub(r'--', '', without_punc)
        without_punc = re.sub(r'\s\s', ' ', without_punc).rstrip(' ')  # remove double spaces and space at the end
        return without_punc

    def get_tokens(self):
        return [token.text for token in self.sentence]

    def split_into_clauses(self):
        """Split a sentence into clauses.
        Clauses are seperated by interpunctuation or a conjunction
        Problem: People didn't always write the sentences well so these rules might not apply."""
        # TODO
        pass

    def extract_main_clause(self):
        # TODO
        pass



# ---------------------------------- example ---------------------------------
# TODO delete examples when preprocessing is finished
sentence1 = 'Schliesslich hinterfragt der dritte Teil, welche Verhaltensweisen der Lehrpersonen und Schüler/innen in den ' \
            'Romanen sich als vertrauensförderlich bzw. –hinderlich sind.'
sentence2 = 'Ein Bild ist zwar etwas völlig anderes als ein Text, doch steht es selten allein.'
sentence3 = 'Ob eine oder mehrere Thesen zutreffen werden, werde ich in meiner Arbeit untersuchen, unter der ' \
            'Fragestellung, wie ein Opfer zum Opfer wird.'
s1 = Sentence(sentence1)
s2 = Sentence(sentence2)
s3 = Sentence(sentence3)

# Case 1: lemmatize; remove punc and stopwords from entire sentence
print(s1.sentence)
print(s1.preprocessed)

# TODO: Case 2.1.: Extract main clauses

# TODO: Case 2.1.: lemmatize, remove punc from extracted main clause