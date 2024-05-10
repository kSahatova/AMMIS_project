from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import cpath


clf_datasets = [
    ("breast-cancer", "breast_cancer", "imodels")
]

# Read in data set
# get_clean_dataset is not presented in the repo provided by the authors
# todo: Rewrite data extraction
X, y, feature_names = get_clean_dataset('breast_cancer', data_source='imodels')

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# number of trees
ntrees = 10

clf = RandomForestClassifier(n_estimators=ntrees)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

P = cpath(clf, X_test, y_test)
T = cpath.transition(P, X_test, y_test)
Imp = cpath.importance(T)

Imp["global"] # global explanations
Imp["local"] # local explanations
