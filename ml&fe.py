import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score  
df=pd.read_csv("payment_fraud.xls")
df.head()
df = pd.get_dummies(df, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('label', axis=1), df['label'], 
    test_size=0.33, random_state=17
)
clf = LogisticRegression(max_iter=1000) 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
