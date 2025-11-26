import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np

california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Ridge RMSE:", rmse)


with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
