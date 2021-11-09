import re
import string

import nltk
import pandas as pd
from imblearn.over_sampling import SMOTE
from nltk.tokenize.casual import casual_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


class DiplomacyMessageVectorizer:
    training_matrix = None
    training_labels = None

    def __init__(self):
        print('-- DiplomacyMessageVectorizer init --')
        self._stopwords = nltk.corpus.stopwords.words('english')
        self._punctuations = string.punctuation + '\’\”'
        self.vectorizer = self._create_vectorizer()

    def _create_vectorizer(self):
        vectorizer = TfidfVectorizer(
                        max_df=0.90,
                        max_features=100000,
                        min_df=0.05,
                        stop_words=self._stopwords,
                        use_idf=True,
                        tokenizer=self._tokenize_and_remove_stopwords,
                        ngram_range=(1,3)
                    )
        return vectorizer

    def fit_transform(self, train_df, oversample=True):
        assert 'messages' in train_df.columns, 'messages column not found'
        assert 'sender_labels' in train_df.columns, 'sender_labels not found'

        train_df = self._preprocess_messages(train_df)
        self.training_labels = train_df['sender_labels'].astype(int)
        self.training_matrix = (self.vectorizer
                                    .fit_transform(train_df['messages'])
                                    .todense())

        if oversample:
            self._oversample_training_minority()

        return self.training_matrix, self.training_labels

    def transform(self, df):
        df['messages'] = df['messages'].apply(lambda x: self.clean_message(x))
        return self.vectorizer.transform(df['messages']).todense()

    def _oversample_training_minority(self):
        oversampler = SMOTE(sampling_strategy='minority', k_neighbors=5)

        self.training_matrix, self.training_labels = (
            oversampler.fit_resample(self.training_matrix,
                                     self.training_labels.ravel())
        )

    def _tokenize_and_remove_stopwords(self, message):
        tokens = casual_tokenize(message, reduce_len=True)
        tokens = ([t for t in tokens
                        if t not in self._stopwords and
                           t not in self._punctuations
                 ])
        return tokens

    @classmethod
    def _preprocess_messages(cls, df):
        df['messages'] = df['messages'].apply(lambda x: cls.clean_message(x))
        df = df.drop(df.loc[df['messages'] == ''].index)
        return df

    @staticmethod
    def remove_emojis(message):
        emoji_pattern = re.compile(
                    pattern = u"[\U0001F600-\U0001F64F"     # emoticons
                                "\U0001F300-\U0001F5FF"     # symbols & pictographs
                                "\U0001F680-\U0001F6FF"     # transport & map symbols
                                "\U0001F1E0-\U0001FAD6]+",  # flags (iOS)
                    flags = re.UNICODE)
        return emoji_pattern.sub(r'', message)

    @classmethod
    def clean_message(cls, message):
        message = (message.replace('\n', ' ')
                          .replace('-', ' - ')
                          .replace('...', ' ')
                          .replace('???', '?')
                  )
        message = cls.remove_emojis(message)

        return message.lower().strip()
