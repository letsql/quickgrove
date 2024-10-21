import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

import kagglehub

path = kagglehub.dataset_download("teejmahal20/airline-passenger-satisfaction")

print("Path to dataset files:", path) 

df = pd.read_csv(path+"/train.csv")

df = df.drop(['id', 'Unnamed: 0'], axis=1)
df = df.dropna()
df.columns = df.columns.str.replace(' ', '_').str.lower()
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('satisfaction', axis=1)
y = df['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'num_parallel_tree': 10,  # This will create 10 trees per boosting round
}

# Train the model
num_boost_round = 150  # This will result in 1500 trees (10 * 150)
model = xgb.train(params, dtrain, num_boost_round)

# Make predictions
y_pred = model.predict(dtest)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

# Save the model
model.save_model('large-xgboost-model.json')
X_test.columns = X_test.columns.str.replace(' ', '_').str.lower()
X_test.to_csv('airline-passenger-satisfaction.csv', index=False)
print(f"Number of trees in the model: {model.num_boosted_rounds() * params['num_parallel_tree']}")
