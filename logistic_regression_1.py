import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./datasets/advanced_housing_data.csv")

df['Price_Category'] = (df['Price_USD'] > df['Price_USD'].median()).astype(int)

plt.figure(figsize=(10,6))
sns.scatterplot(
    x="Square_Feet",
    y="Price_USD",
    hue="Price_Category",
    data=df,
    alpha=0.6
)
plt.title('Square Feet vs Price (with Category)')
plt.xlabel('Square Feet')
plt.ylabel('Price USD')
plt.show()

features = ['Square_Feet', 'House_Age']
label = 'Price_Category'

X = df[features]
y = df[label]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=35
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print('Square_Feet\tHouse_Age\tActual\tPredicted')

for i in range(10):
    print(f"{X_test.iloc[i]['Square_Feet']}\t"
          f"{X_test.iloc[i]['House_Age']}\t"
          f"{Y_test.iloc[i]}\t"
          f"{y_pred[i]}")

from sklearn import metrics

cm = metrics.confusion_matrix(Y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("--- Model Performance ---")
print(f"Accuracy: {metrics.accuracy_score(Y_test, y_pred):.2f}")
print(f"Precision: {metrics.precision_score(Y_test, y_pred):.2f}")
print(f"Recall: {metrics.recall_score(Y_test, y_pred):.2f}")
print(f"F1 Score: {metrics.f1_score(Y_test, y_pred):.2f}")