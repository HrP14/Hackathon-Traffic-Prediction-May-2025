##Importing Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

##Loading and Parsing the Dataset

train_df = pd.read_csv("/kaggle/input/123-of-ai-presents-traffic-prediction-may-2025/train_set_dirty.csv")
test_df = pd.read_csv("/kaggle/input/123-of-ai-presents-traffic-prediction-may-2025/test_set_nogt.csv")
train_df['date_time'] = pd.to_datetime(train_df['date_time'], dayfirst=True, errors='coerce')
test_df['date_time'] = pd.to_datetime(test_df['date_time'], dayfirst=True, errors='coerce')

"""
We read the data using pandas.read_csv().

We convert the date_time column to a proper datetime object using pd.to_datetime() so we can extract features like hour, day, and month.

dayfirst=True tells pandas to interpret the format like 13-10-2012 as October 13 (European style).

errors='coerce' converts bad/missing formats to NaT (missing value).
"""

##Handling Missing Target Values

train_df = train_df.dropna(subset=['traffic_volume'])

"""
We remove rows that don’t have the target value (traffic_volume) because we can’t use them for training.
"""

##Feature Engineering on DateTime

def enrich_time_features(df):
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df

train_df = enrich_time_features(train_df)
test_df = enrich_time_features(test_df)

"""
We create new features from the timestamp because they help the model learn patterns:

    hour: Rush hours matter for traffic.

    dayofweek: Monday vs Sunday traffic differs.

    month: Seasonal patterns.

    is_weekend: Binary flag to indicate weekends, which generally have lower traffic.
"""

##Selecting Features and Target

features = ['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all',
            'weather_main', 'weather_description', 'hour', 'dayofweek', 'month', 'is_weekend']
X = train_df[features]
y = train_df['traffic_volume']
X_test = test_df[features]

"""
We manually define what features go into the model.

X is the training input, y is the target.

X_test is the test set where we’ll make predictions later.
"""

##Train-Test Split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

"""
Splits the training data into 80% training and 20% validation.
"""

##Preprocessing Setup

numeric_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'dayofweek', 'month', 'is_weekend']
categorical_features = ['holiday', 'weather_main', 'weather_description']
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

"""
We handle different types of features:

    Numeric: Missing values filled with the mean.

    Categorical: Missing values filled with the most frequent value, then one-hot encoded to turn text labels into numbers.

Since ML models only work with numbers and don’t like missing data, so we must convert everything into clean numeric arrays.
"""

##Baseline Model: Linear Regression

baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

baseline_pipeline.fit(X_train, y_train)
y_pred_base = baseline_pipeline.predict(X_val)

##Baseline Model: Evaluation

print("Baseline Model (Linear Regression):")
print("RMSE:", mean_squared_error(y_val, y_pred_base, squared=False))
print("MAE:", mean_absolute_error(y_val, y_pred_base))

"""
A simple, fast model assuming a straight-line relationship between features and target.

Helps establish a baseline RMSE/MAE so we know what to beat.
"""

##Improved Model: Random Forest Regressor 

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_val)

##Improved Model: Evaluation

print("\nRandom Forest Model:")
print("RMSE:", mean_squared_error(y_val, y_pred_rf, squared=False))
print("MAE:", mean_absolute_error(y_val, y_pred_rf))

"""
Random Forest = collection of decision trees.

Handles nonlinear relationships and interactions well.

It averages the predictions from many trees to reduce overfitting and variance.

We use n_estimators=100 (100 trees) and max_depth=15 to limit complexity
"""

##Model Evaluation (Repeat)


print("Baseline Model (Linear Regression):")
print("RMSE:", mean_squared_error(y_val, y_pred_base, squared=False))
print("MAE:", mean_absolute_error(y_val, y_pred_base))
print("\nRandom Forest Model:")
print("RMSE:", mean_squared_error(y_val, y_pred_rf, squared=False))
print("MAE:", mean_absolute_error(y_val, y_pred_rf))

"""
RMSE (Root Mean Squared Error): Measures average prediction error

MAE (Mean Absolute Error): Measures average absolute difference between prediction and actual.

Both are important — lower values = better model.
"""

##Test Prediction & Submission File

submission = pd.DataFrame({
    'ID': range(len(test_predictions)),
    'traffic_volume': test_predictions
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("\n Submission file 'submission.csv' created.")

"""
Predicts traffic volume for the test data.

Prepares a submission file in the required format (Id, traffic_volume).
"""