import pickle
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from vars import Paths


def get_data():
    le = LabelEncoder()
    df = pd.read_csv(Paths.URLS.value, na_filter=False)
    x, y = df.content, df.category

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y)

    y_train_lb = le.fit_transform(y_train.values.tolist())
    y_test_lb = le.transform(y_test.values.tolist())

    return x_train, x_test, y_train_lb, y_test_lb, le


def train_model(clfs):
    x_train, x_test, y_train, y_test, le = get_data()
    init_accuracy = 0

    for name, clf in clfs:
        classifier = Pipeline([
            ('count', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', clf)])

        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test)

        accuracy = accuracy_score(y_test, pred)

        print(accuracy)
        print(classification_report(y_test, pred))

        if accuracy > init_accuracy:
            best_clf = {'clf': classifier, 'name': name}
            init_accuracy = accuracy

    pickle.dump({'le': le, 'clf': best_clf['clf']}, open(
        Paths.CATEGORIES_MODEL.value, 'wb'))
    print('Best classifier is', best_clf['name'])


if __name__ == '__main__':
    tic = time.perf_counter()
    classifiers = [
        ('RandomForest', RandomForestClassifier()),
        ('KNeighbors', KNeighborsClassifier()),
        ('ComplementNB', ComplementNB()),
    ]
    train_model(classifiers)
    print(f'Trained in {round(time.perf_counter() - tic, 2)} seconds')
