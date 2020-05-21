from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix


def save_confusion_matrix(cm, name):
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cmn, annot=True, cmap=plt.cm.Blues, cbar=False, annot_kws={"size": 16})
    plt.title(f'{name} confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(Path('./graphs', f'{name}_cm.png'))
    plt.close()


def save_bars(y, y_train, y_test, name):
    y.value_counts().plot(figsize=(12, 8), kind='bar')
    plt.xlabel('Context')
    plt.ylabel('Number of elements')
    plt.tight_layout()
    plt.savefig(Path('./graphs', f'all_{name}.png'))
    plt.close()

    y_train.value_counts().plot(figsize=(12, 8), kind='bar')
    plt.xlabel('Context')
    plt.ylabel('Number of elements')
    plt.tight_layout()
    plt.savefig(Path('./graphs', f'train_{name}.png'))
    plt.close()

    y_test.value_counts().plot(figsize=(12, 8), kind='bar')
    plt.xlabel('Context')
    plt.ylabel('Number of elements')
    plt.tight_layout()
    plt.savefig(Path('./graphs', f'test_{name}.png'))
    plt.close()
