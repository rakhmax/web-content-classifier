import pickle
import time
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from vars import Paths


def get_data():
    df = pd.read_csv(Paths.URLS.value, na_filter=False)
    x = df.content
    y = []

    for col in df.iloc[:, 3:].columns:
        df[col] = df[col].apply(lambda x: col if x == 1 else '')

    for yi in df.iloc[:, 3:].to_numpy():
        y.append([i for i in yi if i])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3)

    mlb = MultiLabelBinarizer()
    y_train_mlb = mlb.fit_transform(y_train)
    y_test_mlb = mlb.transform(y_test)

    return x_train, x_test, y_train_mlb, y_test_mlb, mlb


def train_model(clfs):
    x_train, x_test, y_train, y_test, mlb = get_data()
    init_accuracy = 0

    for name, clf in clfs:
        classifier = Pipeline([
            ('count', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', OneVsRestClassifier(clf))])

        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test)

        accuracy = accuracy_score(y_test, pred)

        print(accuracy)
        print(classification_report(y_test, pred))

        if accuracy > init_accuracy:
            best_clf = {'clf': classifier, 'name': name}
            init_accuracy = accuracy

    pickle.dump({'mlb': mlb, 'clf': best_clf['clf']}, open(
        Paths.CTX_MODEL.value, 'wb'))
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
