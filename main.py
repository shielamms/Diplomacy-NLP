import classifier as clf
import pandas as pd
import reader
import vectorizer as vct
from sklearn.metrics import classification_report

def evaluate(X, y, predictions):
    eval_report = classification_report(list(y), predictions)
    eval_results = pd.DataFrame({
        'Messages': X,
        'Prediction': list(predictions),
        'Actual': y,
    })

    print('Classification Report:\n', eval_report)
    print('=' * 20, '\n')

    return eval_results


if __name__ == '__main__':
    # Read the training data file
    reader = reader.DiplomacyGamesReader()
    training_df = reader.read_from_file('data/train.jsonl')

    # Vectorize the training messages
    vectorizer = vct.DiplomacyMessageVectorizer()
    X_train, y_train = vectorizer.fit_transform(training_df)

    # Train the classifier with the vectorized training messages
    classifier = clf.DiplomacyMessageClassifier('SVC')
    classifier.train(X_train, y_train)

    # Validate
    validation_df = reader.read_from_file('data/validation.jsonl')
    X_val = vectorizer.transform(validation_df)
    predictions = classifier.predict(X_val)

    print(f'Validation Results: (model: {classifier.name})')
    evaluate(validation_df['messages'],
             validation_df['sender_labels'],
             predictions
            )

    # Test
    test_df = reader.read_from_file('data/test.jsonl')
    X_test = vectorizer.transform(test_df)
    predictions = classifier.predict(X_test)

    print(f'Test Results: (model: {classifier.name})')
    evaluate(test_df['messages'],
             test_df['sender_labels'],
             predictions
            )

    # Test input
    while True:
        print('\n')
        test_message = input('Your test message: ')
        X_test = vectorizer.transform(
                    pd.DataFrame({'messages': [test_message]})
                )
        prediction = classifier.predict(X_test)[0]

        print('The AI thinks...')

        if prediction:
            print('That is a truth!')
        else:
            print('That is a despicable lie!')
