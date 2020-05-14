import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from vars import Paths


def prepare_categories_dataset():
    df = pd.read_csv(Paths.CATEGORIES.value, na_filter=False)
    x, y = df.category, df[['common', 'kid', 'office', 'student']]

    tfidf = TfidfVectorizer()
    tfidf.fit(x)
    x = tfidf.transform(x)

    for user in y:
        clf = ComplementNB()
        clf.fit(x, y[user])
        pred = clf.predict(x)
        print(clf.predict(x))
        print(accuracy_score(y[user], pred))


def prepare_urls_dataset():
    df = pd.read_csv(Paths.URLS.value, na_filter=False)
    x, y = df.content, df.category

    tfidf = TfidfVectorizer()
    tfidf.fit(x)
    tfidf.transform(x)

    pickle.dump(tfidf, open(Paths.FEATURES.value, 'wb'))

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    x_train_tfidf = tfidf.transform(x_train)
    x_test_tfidf = tfidf.transform(x_test)

    return x_train_tfidf, x_test_tfidf, y_train, y_test


def save_best_model(clfs):
    x_train, x_test, y_train, y_test = prepare_urls_dataset()
    init_accuracy = 0

    for name, clf in clfs:
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

        accuracy = accuracy_score(y_test, pred)

        print(f'\n{name}:')
        print(pred)
        print(accuracy)
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred))

        if accuracy > init_accuracy:
            best_clf = clf
            init_accuracy = accuracy

    pickle.dump(best_clf, open(Paths.MODEL.value, 'wb'))
    print('Best classifier is', best_clf)


if __name__ == '__main__':
    classifiers = [
        ('RandomForest', RandomForestClassifier()),
        ('KNeighbors', KNeighborsClassifier()),
        ('ComplementNB', ComplementNB())
    ]
    prepare_categories_dataset()
    save_best_model(classifiers)
