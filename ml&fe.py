# Machine Learning and Feature Engineering
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

# Data visualisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score  

# Load the dataset
df = pd.read_csv("payment_fraud.xls")
df.head()

# Preprocess the dataset
df = pd.get_dummies(df, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('label', axis=1), df['label'], 
    test_size=0.33, random_state=17
)

# Create and train the model
clf = LogisticRegression(max_iter=1000) 
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Print results
print("Confusion Matrix:\n", cm)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Fraud', 'Fraud'], 
            yticklabels=['Not Fraud', 'Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Create a pie chart for the distribution of actual classes
plt.figure(figsize=(8, 6))
labels = ['Not Fraud', 'Fraud']
sizes = [np.sum(y_test == 0), np.sum(y_test == 1)]  # Count of each class in the test set
colors = ['#66c2a5', '#fc8d62']  # Colors for the pie chart
explode = (0.1, 0)  # Only "explode" the 1st slice (Not Fraud)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Distribution of Actual Classes in Test Set')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular
plt.show()
