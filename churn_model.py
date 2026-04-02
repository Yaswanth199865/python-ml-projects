import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'age': [22, 25, 47, 52, 46, 56],
    'salary': [15000, 29000, 48000, 60000, 52000, 75000],
    'churn': [0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['age', 'salary']]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
