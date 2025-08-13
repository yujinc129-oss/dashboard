import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# CSV 읽기
df = pd.read_csv("drama_dataset.csv")

X = df.drop(columns=["rating"])
y = df["rating"]

cat_features = ["actor1", "actor2", "genre", "platform"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
], remainder="drop")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "drama_rating_model.pkl")
print("✅ 모델 학습 및 저장 완료!")
