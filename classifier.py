import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from vars import Paths


def split_dataset():
    df = pd.read_csv(Paths.URLS.value, na_filter=False)
    x, y = df.content, df.category

    tfidf = TfidfVectorizer(max_features=1500, lowercase=False)
    tfidf.fit_transform(x)

    pickle.dump(tfidf, open(Paths.FEATURES.value, 'wb'))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y)

    x_train_tfidf = tfidf.transform(x_train)
    x_test_tfidf = tfidf.transform(x_test)

    return x_train_tfidf, x_test_tfidf, y_train, y_test


def train_model(clfs):
    x_train, x_test, y_train, y_test = split_dataset()
    init_accuracy = 0

    y_train.value_counts().plot(figsize=(14, 10), kind='bar', color='grey')
    plt.savefig('trainStratify.png')

    for name, clf in clfs:
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

        accuracy = accuracy_score(y_test, pred)

        print(f'\n{name}:')
        print(accuracy)
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred))

        if accuracy > init_accuracy:
            best_clf = clf
            init_accuracy = accuracy

    pickle.dump(best_clf, open(Paths.MODEL.value, 'wb'))
    print('Best classifier is', best_clf)


if __name__ == '__main__':
    tic = time.perf_counter()
    classifiers = [
        ('RandomForest', RandomForestClassifier()),
        ('KNeighbors', KNeighborsClassifier()),
        ('ComplementNB', ComplementNB()),
    ]
    train_model(classifiers)
    print(f'Classified in {round(time.perf_counter() - tic, 2)} seconds')
