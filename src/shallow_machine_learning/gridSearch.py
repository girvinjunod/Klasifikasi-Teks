# Grid Search for best Params

from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

maps = {"no": 0, "yes": 1}

path = "../../data_worthcheck/"
data_train = pd.read_csv(path + "train.csv")

vectorizer = CountVectorizer(
    token_pattern=r"[A-Za-z_][A-Za-z\d_]*", min_df=0.001)
X_train = vectorizer.fit_transform(data_train.text_a).toarray()
y_train = data_train["label"]

y_train = y_train.replace(maps)

model = XGBClassifier()
parameters = {'n_estimators': [100, 500],
              'learning_rate': [0.1, 0.3],
              'tree_method': ['hist', 'approx', 'exact']}


clf = GridSearchCV(model, parameters, n_jobs=-1)

clf.fit(X_train, y_train)

model_params = clf.best_params_
print(model_params)
