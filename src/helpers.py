import nltk

from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import metrics

nltk.download('punkt')
nltk.download('stopwords')


class PorterTokenizer:

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        self.stopwords.add("'")

    def __call__(self, doc):
        return [self.stemmer.stem(token) for token in wordpunct_tokenize(doc) if token not in self.stopwords]


def points2group(row):
    """Convert points to it's group according to WineMag"""
    if 80 <= row['points'] <= 82:
        return '80-82 Classic The pinnacle of quality.'
    if 83 <= row['points'] <= 86:
        return '83-86 Superb A great achievement.'
    if 87 <= row['points'] <= 89:
        return '87-89 Excellent Highly recommended.'
    if 90 <= row['points'] <= 93:
        return '90-93 Very Good Often good value; well recommended.'
    if 94 <= row['points'] <= 97:
        return '94-97 Good Suitable for everyday consumption; often good value.'
    if 98 <= row['points'] <= 100:
        return '98-100'
    return 'UNKNOWN'


def evaluate_reg(reg, X, y):

    predicted = reg.predict(X)

    print('Mean squared error:')
    print(metrics.mean_squared_error(y, predicted))
    print()

    print('Mean absolute error:')
    print(metrics.mean_absolute_error(y, predicted))
    print()

    print('R^2 score function:')
    print(metrics.r2_score(y, predicted))
    print()


def evaluate_clf(clf, X, y, labels=None):

    predicted = clf.predict(X)

    print('Confusion matrix:')
    print(metrics.confusion_matrix(y, predicted, labels=labels))
    print()

    print('Accuracy:')
    print(metrics.accuracy_score(y, predicted))
    print()

    print('Precision per class:')
    print(metrics.precision_score(y, predicted, average=None, labels=labels))
    print()

    print('Recall per class:')
    print(metrics.recall_score(y, predicted, average=None, labels=labels))
    print()

    print('F1 per class:')
    print(metrics.f1_score(y, predicted, average=None, labels=labels))
    print()
