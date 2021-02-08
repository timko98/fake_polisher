from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score

from src.prepare_datasets import prepare_datasets


def main():
    """
    RidgeRegression classifier.
    """
    dct = True
    X_train, X_test, y_train, y_test = prepare_datasets(dct)
    model = RidgeClassifier(alpha=1.0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, pred)
    print(f"Balanced accuracy score: {bal_acc:g}")


if __name__ == '__main__':
    main()
