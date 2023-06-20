import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Loading the training and test data
train_data = pd.read_csv('cybersecurity_training.csv', delimiter='|')
test_data = pd.read_csv('cybersecurity_test.csv', delimiter='|')

# Selecting relevant columns as features
features = ['overallseverity', 'timestamp_dist', 'correlatedcount', 'score',
            'isiptrusted', 'untrustscore', 'flowscore', 'trustscore', 'enforcementscore']
features2 = ['overallseverity', 'timestamp_dist', 'correlatedcount', 'score',
             'isiptrusted', 'untrustscore', 'flowscore', 'trustscore', 'enforcementscore',
             'srcipcategory_cd', 'dstipcategory_cd', 'thrcnt_month', 'thrcnt_week',
             'thrcnt_day', 'reportingdevice_cd']
features3 = ['overallseverity', 'timestamp_dist', 'correlatedcount', 'score',
             'isiptrusted', 'untrustscore', 'flowscore', 'trustscore', 'enforcementscore',
             'srcipcategory_cd', 'dstipcategory_cd', 'alerttype_cd', 'eventname_cd',
             'reportingdevice_cd', 'severity_cd']

# Extracting the target variable from the training data
y_train = train_data['notified']

# Creating RandomForestClassifier models with different configurations
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model2 = RandomForestClassifier(n_estimators=1000, random_state=42)
model3 = RandomForestClassifier(n_estimators=100, random_state=42)
model4 = RandomForestClassifier(n_estimators=100, random_state=42)

# Handling missing values in the 'score' column
mean_value = train_data['score'].mean()
train_data['score'].fillna(mean_value, inplace=True)
mean_value = test_data['score'].mean()
test_data['score'].fillna(mean_value, inplace=True)

# Training the models
X1_train = train_data[features]
model1.fit(X1_train, y_train)

X2_train = train_data[features]
model2.fit(X2_train, y_train)

X3_train = train_data[features2]
model3.fit(X3_train, y_train)

X4_train = train_data[features3]
model4.fit(X4_train, y_train)

# Predicting on the validation data
X1_val = test_data[features]
y1_pred = model1.predict_proba(X1_val)[:, 1]

X2_val = test_data[features]
y2_pred = model2.predict_proba(X2_val)[:, 1]

X3_val = test_data[features2]
y3_pred = model3.predict_proba(X3_val)[:, 1]

X4_val = test_data[features3]
y4_pred = model4.predict_proba(X4_val)[:, 1]

# Saving the results to separate files
with open('ResultsRandomForest1.txt', 'w') as file:
    for result in y1_pred:
        file.write(str(result) + '\n')

with open('ResultsRandomForest2.txt', 'w') as file:
    for result in y2_pred:
        file.write(str(result) + '\n')

with open('ResultsRandomForest3.txt', 'w') as file:
    for result in y3_pred:
        file.write(str(result) + '\n')

with open('ResultsRandomForest4.txt', 'w') as file:
    for result in y4_pred:
        file.write(str(result) + '\n')
