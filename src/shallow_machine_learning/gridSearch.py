# Grid Search for best Params

from xgboost import XGBClassifier
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

maps = {"no": 0, "yes": 1}

path = "../../data_worthcheck/"
data_train = pd.read_csv(path + "train.csv")

vectorizer = CountVectorizer(
    token_pattern=r"[A-Za-z_][A-Za-z\d_]*", min_df=0.001)
X_train = vectorizer.fit_transform(data_train.text_a).toarray()
y_train = data_train["label"]

model = XGBClassifier(tree_method='hist')
parameters = {'n_estimators': [5, 100, 300, 500, 1000],
              'learning_rate': [0.05, 0.1, 0.2, 0.3],
              'max_depth': [6, 12, 18, 24],
              'subsample': [0, 5, 0, 67, 0.8]}


clf = GridSearchCV(model, parameters, n_jobs=-1)

clf.fit(X_train, y_train)

model_params = clf.best_params_
print(model_params)
