import xgboost as xgb
import numpy as np
import pandas as pd

n_samples = 1000
np.random.seed(42)

continuous = np.random.normal(0, 1, n_samples)

integer = np.random.randint(0, 10, n_samples)

quantized = np.random.choice([0.0, 0.25, 0.5, 0.75, 1.0], n_samples)

categories = ['A', 'B', 'C', 'D']
categorical = np.random.choice(categories, n_samples)

df = pd.DataFrame({
    'continuous': continuous,
    'integer': integer,
    'quantized': quantized,
    'categorical': categorical
})

for category in categories:
    df[f'cat_{category}'] = (df['categorical'] == category).astype(int)

df = df.drop('categorical', axis=1)

y = (2 * continuous + 
     3 * integer + 
     4 * quantized + 
     5 * df['cat_A'] + 
     6 * df['cat_B'] + 
     7 * df['cat_C'] + 
     8 * df['cat_D'] + 
     np.random.normal(0, 0.1, n_samples))

dtrain = xgb.DMatrix(df, label=y)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'eta': 0.1,
}

model = xgb.train(params, dtrain)

model.save_model('reg_squarederror.json')
df.to_csv('xgboost_test_data.csv', index=False)
print("Feature names:", list(df.columns))
