import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix


def save_confusion_matrix(cm, name):
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cmn, annot=True, cmap=plt.cm.Blues)
    plt.title(f'{name} confusion matrix')
    plt.savefig(f'{name}_cm.png')
    plt.close()


def save_bars(y, y_train, y_test, name):
    y.value_counts().plot(figsize=(12, 8), kind='bar')
    plt.xlabel('Категории')
    plt.ylabel('Кол-во элементов')
    plt.tight_layout()
    plt.savefig(f'all_{name}.png')
    plt.close()

    y_train.value_counts().plot(figsize=(12, 8), kind='bar')
    plt.xlabel('Категории')
    plt.ylabel('Кол-во элементов')
    plt.tight_layout()
    plt.savefig(f'train_{name}.png')
    plt.close()

    y_test.value_counts().plot(figsize=(12, 8), kind='bar')
    plt.xlabel('Категории')
    plt.ylabel('Кол-во элементов')
    plt.tight_layout()
    plt.savefig(f'test_{name}.png')
    plt.close()
