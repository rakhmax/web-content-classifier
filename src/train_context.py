import pickle
import time
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.svm import LinearSVC
from reports import save_bars, save_confusion_matrix
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
        x, y, test_size=0.3, stratify=y)

    y_plot = pd.Series(np.concatenate(y))
    y_train_plot = pd.Series(np.concatenate(y_train))
    y_test_plot = pd.Series(np.concatenate(y_test))

    save_bars(y_plot, y_train_plot, y_test_plot, 'context')

    mlb = MultiLabelBinarizer()
    y_train_mlb = mlb.fit_transform(y_train)
    y_test_mlb = mlb.transform(y_test)

    return x_train, x_test, y_train_mlb, y_test_mlb, mlb


def train_model(clfs):
    x_train, x_test, y_train, y_test, mlb = get_data()
    init_recall = 0

    for name, clf in clfs:
        classifier = Pipeline([
            ('count', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', OneVsRestClassifier(clf))])

        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test)

        for i, cm in enumerate(multilabel_confusion_matrix(y_test, pred)):
            save_confusion_matrix(cm, f'{name}_{i}')

        accuracy = accuracy_score(y_test, pred)
        recall = recall_score(y_test, pred, average='micro')

        print(name)
        print('-----------------------------------------------------')
        print(f'accuracy score: {round(accuracy, 2)}\n')
        print(classification_report(y_test, pred, zero_division=0))

        if recall > init_recall:
            best_clf = {'clf': classifier, 'name': name}
            init_recall = recall

    pickle.dump({'mlb': mlb, 'clf': best_clf['clf']}, open(
        Paths.CTX_MODEL.value, 'wb'))
    print('Best classifier is', best_clf['name'])


if __name__ == '__main__':
    tic = time.perf_counter()
    classifiers = [
        ('OvR Random Forest', RandomForestClassifier()),
        ('OvR K-Nearest', KNeighborsClassifier()),
        ('OvR Complement Naive Bayes', ComplementNB()),
        ('OvR Logistic Regression', LogisticRegression()),
        ('OvR Linear SVC', LinearSVC())
    ]
    train_model(classifiers)
    print(f'Trained in {round(time.perf_counter() - tic, 2)} seconds')
