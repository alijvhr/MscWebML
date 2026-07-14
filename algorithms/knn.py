from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def create_knn(is_classification: bool = True):
    if is_classification:
        return KNeighborsClassifier(n_neighbors=3)
    return KNeighborsRegressor(n_neighbors=3)
