#Numerical
import numpy as np
import pandas as pd

#Date and Time
from datetime import datetime, timedelta

#Label Encoding
from sklearn.preprocessing import LabelEncoder

#Model
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Hyperparameter tuning
import optuna

# Main training and test data
train = pd.read_csv('train.csv', parse_dates=['date'])
test = pd.read_csv('test.csv', parse_dates=['date'])

# Static store information
stores = pd.read_csv('stores.csv')

# Daily transaction counts
transactions = pd.read_csv('transactions.csv', parse_dates=['date'])

# Daily oil prices
oil = pd.read_csv('oil.csv', parse_dates=['date'])

# Holidays and events
holidays = pd.read_csv('holidays_events.csv', parse_dates=['date'])

# Rename columns in holidays to avoid conflict with store 'type'
holidays = holidays[holidays['transferred'] == False]
holidays_simple = holidays[['date', 'type', 'locale']].drop_duplicates()
holidays_simple = holidays_simple.rename(columns={
    'type': 'holiday_type',
    'locale': 'holiday_locale'
})

# Merge store information (type, cluster, etc.)
train = train.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')

# Merge daily transaction counts
train = train.merge(transactions, on=['date', 'store_nbr'], how='left')

# Rename column for clarity before merge
oil = oil.rename(columns={'dcoilwtico': 'oil_price'})
train = train.merge(oil, on='date', how='left')
test = test.merge(oil, on='date', how='left')

# Merge holiday info
train = train.merge(holidays_simple, on='date', how='left')
test = test.merge(holidays_simple, on='date', how='left')

# Fill missing transactions with forward fill by store
train.sort_values(['store_nbr', 'date'], inplace=True)
train['transactions'] = train.groupby('store_nbr')['transactions'].ffill()

# Fill oil prices with forward fill
train['oil_price'] = train['oil_price'].ffill()
test['oil_price'] = test['oil_price'].ffill()

# Fill missing holiday info with 'NoHoliday'
train['holiday_type'] = train['holiday_type'].fillna('NoHoliday')
test['holiday_type'] = test['holiday_type'].fillna('NoHoliday')

train['holiday_locale'] = train['holiday_locale'].fillna('NoLocale')
test['holiday_locale'] = test['holiday_locale'].fillna('NoLocale')

#Feature Engineering
def create_features(df):
    # Date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_month_mid'] = df['day'].between(13, 18).astype(int)

    # Holiday features
    df['is_holiday'] = df['date'].isin(holiday_dates).astype(int)
    df['is_holiday_1day_ahead'] = df['date'].add(pd.Timedelta(days=1)).isin(holiday_dates).astype(int)
    df['is_holiday_2days_ahead'] = df['date'].add(pd.Timedelta(days=2)).isin(holiday_dates).astype(int)

    return df

def create_lag_features(df, lags=[1, 7]):
    df = df.sort_values(by=['store_nbr', 'family', 'date'])
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(lag)
    return df



# Tatil tarihlerini önce tanımlayalım (fonksiyon dışında)
holiday_dates = holidays_simple['date'].unique()

# Uygulama
train = create_features(train)
test = create_features(test)

train = create_lag_features(train, lags=[1, 7])
train[['sales_lag_1', 'sales_lag_7']] = train[['sales_lag_1', 'sales_lag_7']].fillna(0)


# Label Encoding
cat_cols = [
    'store_nbr', 'family', 'city', 'state', 'type', 'cluster',
    'holiday_type', 'holiday_locale'
]

# Apply Label Encoding on both train and test
for col in cat_cols:
    le = LabelEncoder()
    all_values = pd.concat([train[col], test[col]], axis=0).astype(str)  # combine to avoid mismatch
    le.fit(all_values)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Drop unnecessary columns
target = 'sales'
drop_cols = ['id', 'date', 'sales']
# Only keep columns present in both train and test
features = [col for col in train.columns if col not in drop_cols and col in test.columns]

# Optionally, include lag features ONLY if they exist in test set
lag_cols = ['sales_lag_1', 'sales_lag_7']
for col in lag_cols:
    if col in train.columns and col in test.columns:
        features.append(col)


# Use last 3 months as validation set
cutoff_date = pd.to_datetime('2017-08-01')  # August, September, October

# Split the data based on date
train_set = train[train['date'] < cutoff_date]
val_set = train[train['date'] >= cutoff_date]

# Separate features and target
X_train = train_set[features]
y_train = train_set[target]
X_val = val_set[features]
y_val = val_set[target]

# Define best parameters obtained from Optuna (fixed for final training)
best_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'random_state': 42,
    'learning_rate': 0.04810438517920199,
    'num_leaves': 237,
    'max_depth': 14,
    'min_child_samples': 69,
    'subsample': 0.7940813496689944,
    'colsample_bytree': 0.919752405198694,
    'lambda_l1': 1.9970006537918117,
    'lambda_l2': 2.4390045630647923
}

# Train the LightGBM model using best parameters
model = lgb.train(
    best_params,
    lgb.Dataset(X_train, label=y_train),
    valid_sets=[lgb.Dataset(X_val, label=y_val)],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(20),
        lgb.log_evaluation(period=100)
    ]
)

# Predict and evaluate with RMSLE
y_pred_val = model.predict(X_val)
rmsle_score = np.sqrt(mean_squared_log_error(y_val.clip(0), y_pred_val.clip(0)))
print(f'Validation RMSLE: {rmsle_score:.4f}')

# Predict sales for the test set using the trained model
y_pred_test = model.predict(test[features])

# Create the submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'sales': y_pred_test
})

# Clip negative values (sales can't be negative)
submission['sales'] = submission['sales'].clip(0)

# Save the file for Kaggle submission
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv file has been created successfully.")








