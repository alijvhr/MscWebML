from sklearn.svm import SVC, SVR


def create_svm(is_classification: bool = True):
    if is_classification:
        return SVC()
    return SVR()
