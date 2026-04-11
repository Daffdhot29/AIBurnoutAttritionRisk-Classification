import mlflow 
import pandas as pd 
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier


mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment('Classification-for-Attrition-worker-by-AI')

data = pd.read_csv('../data/ai_worker_burnout_attrition_2026.csv')

# encoder

le = LabelEncoder()
cat_features = data.select_dtypes(include=['object']).columns 

for kol in cat_features : 
    data[kol] = le.fit_transform(data[kol])
data.head()

X = data.drop(columns=['attrition_risk'], axis=1)
y = data['attrition_risk']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

input_example = data[0:10]

with mlflow.start_run() : 
    # log parameters 
    n_estimators = 505
    booster = 'gbtree'
    max_depth = [100,500,5]
    learning_rate = [0.01, 0.2,10]
    gamma = [0,1,5]
    subsample = [0.6,1.0, 5]
    colsample_bytree = [0.6,1,0,5]

    # autolog
    mlflow.autolog()

    # training 
    model = XGBClassifier(n_estimators=n_estimators)
    model.fit(X_train_scaled, y_train)

    # optimalization 
    

