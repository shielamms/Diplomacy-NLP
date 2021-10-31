import json

import nltk
import pandas as pd


class DiplomacyGamesReader:

    TARGET_COLS = ['messages', 'sender_labels', 'receiver_labels',
                   'speakers', 'receivers', 'absolute_message_index',
                   'seasons', 'years', 'game_id']

    def __init__(self):
        print('-- DiplomacyGamesReader init --')
        self.data = None

    def read_from_file(self, filedir):
        games = []
        with open(filedir) as f:
            for line in f:
                game = json.loads(line)
                games.append(game)

        df = pd.DataFrame(games)[self.TARGET_COLS]

        assert 'game_id' in df.columns, "game_id column not found"
        assert 'sender_labels' in df.columns, "sender_labels column not found"

        df = df.set_index('game_id').apply(pd.Series.explode).reset_index()
        df = self.__class__.validate_binary_class_labels(df)
        self.data = df

        return self.data

    @staticmethod
    def validate_binary_class_labels(df):
        # Drop rows without a class label
        df = df.drop(df.loc[df['sender_labels'].isnull()].index)

        if len(df['sender_labels'].unique()) != 2:
            raise AssertionError('Labels are non-binary')

        return df
