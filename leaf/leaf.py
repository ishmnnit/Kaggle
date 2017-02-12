"""Kaggle species leaf identification script."""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]


def classification(x_train, y_train, x_test, y_test):
    """Read train/test data files."""
    for clf in classifiers:
        clf.fit(x_train, y_train)
        name = clf.__class__.__name__

        print("=" * 30)
        print(name)

        train_predictions = clf.predict(x_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Accuracy: {:.4%}".format(acc))

        train_predictions = clf.predict_proba(x_test)
        ll = log_loss(y_test, train_predictions)
        print("Log Loss: {}".format(ll))


def main():
    """Read Train/test log."""
    df = pd.read_csv("train.csv")

    # encode result label
    le = LabelEncoder().fit(df.species)
    labels = le.transform(df.species)
    classes = list(le.classes_)
    print classes

    # drop extra field
    df = df.drop(['species', 'id'], 1)

    # train/test split using stratified sampling
    sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)
    for train_index, test_index in sss:
        x_train, x_test = df.values[train_index], df.values[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

    # classification algorithm
    # classification(x_train, y_train, x_test, y_test)

    # Predict Test Set
    favorite_clf = LinearDiscriminantAnalysis()
    favorite_clf.fit(x_train, y_train)
    test = pd.read_csv('test.csv')
    test_ids = test.id
    test = test.drop(['id'], axis=1)
    test_predictions = favorite_clf.predict_proba(test)
    print test_predictions

    # Format DataFrame
    submission = pd.DataFrame(test_predictions, columns=classes)
    submission.tail()
    submission.insert(0, 'id', test_ids)
    submission.reset_index()
    submission.tail()

    # Export Submission
    submission.to_csv('submission.csv', index=False)
    submission.tail()


if __name__ == '__main__':
    main()
