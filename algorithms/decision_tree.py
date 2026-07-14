from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def create_decision_tree(is_classification: bool = True):
    if is_classification:
        return DecisionTreeClassifier(random_state=1)
    return DecisionTreeRegressor(random_state=1)
