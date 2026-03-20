import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/billy/PycharmProjects/houseanalysis/house-price.csv")
df.head()
df.info()
df.describe()

cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
for col in cols:
    df[col] = df[col].map({'yes':1, 'no':0})

df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
df.isnull().sum()
df.dropna(inplace=True)

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
#INSIGHTS:
# Area ↑ → Price ↑ (strong positive correlation ~0.54)
# Bathrooms has more affect than Bedrooms
# Air conditioning increases the price (~0.45)
# Stories & parking effects the price

plt.scatter(df['area'], df['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()
#possitive affect between area and price
df.groupby('bedrooms')['price'].mean()

from sklearn.model_selection import train_test_split
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, predictions)

