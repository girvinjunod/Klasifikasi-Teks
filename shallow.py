from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
import pandas as pd

# Memakai XGBoost

maps = {"no": 0, "yes": 1}

path = 'data_worthcheck/'
data_train = pd.read_csv(path+'train.csv')
data_test = pd.read_csv(path+'test.csv')


vectorizer = CountVectorizer(
    token_pattern="[A-Za-z_][A-Za-z\d_]*", min_df=0.001)
X_train = vectorizer.fit_transform(data_train.text_a).toarray()
y_train = data_train['label']

X_test = vectorizer.transform(data_test.text_a).toarray()
y_test = data_test['label']

y_train = y_train.replace(maps)
y_test = y_test.replace(maps)
count_df = pd.DataFrame(X_train, columns=vectorizer.get_feature_names_out())

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))
prec = metrics.precision_score(y_test, y_pred)
rec = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
acc = metrics.accuracy_score(y_test, y_pred)
print('Precision Score: {:.4f}'.format(prec))
print('Recall Score: {:.4f}'.format(rec))
print('F1 Score: {:.4f}'.format(f1))
print('Accuracy Score: {:.4f}'.format(acc))
