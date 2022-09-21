import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier

# Memakai XGBoost

maps = {"no": 0, "yes": 1}  # mapping untuk label

path = "../../data_worthcheck/"
data_train = pd.read_csv(path + "train.csv")
data_test = pd.read_csv(path + "test.csv")

# Mengubah data menjadi vector space model
vectorizer = CountVectorizer(
    token_pattern=r"[A-Za-z_][A-Za-z\d_]*", min_df=0.001)
X_train = vectorizer.fit_transform(data_train.text_a).toarray()
X_test = vectorizer.transform(data_test.text_a).toarray()

# Mengambil data label
y_train = data_train["label"]
y_test = data_test["label"]

# Mapping label
y_train = y_train.replace(maps)
y_test = y_test.replace(maps)

# Model XGBoost Classifier
model = XGBClassifier(
    n_estimators=500,
    tree_method="hist",
    subsample=0.8,
    random_state=1,
    eta=0.3,
    max_depth=18,
)

# Fitting Model
model.fit(X_train, y_train)

# Prediksi untuk data test
y_pred = model.predict(X_test)

# Evaluasi Model
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))
prec = metrics.precision_score(y_test, y_pred)
rec = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
acc = metrics.accuracy_score(y_test, y_pred)
print("Precision Score: {:.4f}".format(prec))
print("Recall Score: {:.4f}".format(rec))
print("F1 Score: {:.4f}".format(f1))
print("Accuracy Score: {:.4f}".format(acc))
