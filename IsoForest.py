import pandas as pd
from sklearn.ensemble import IsolationForest

# Loading the training and test data
train_data = pd.read_csv('cybersecurity_training.csv', delimiter='|')
test_data = pd.read_csv('cybersecurity_test.csv', delimiter='|')

# Selecting relevant columns as features
features = ['overallseverity', 'timestamp_dist', 'correlatedcount', 'score',
            'isiptrusted', 'untrustscore', 'flowscore', 'trustscore', 'enforcementscore']


# Handling missing values in the 'score' column
mean_value = train_data['score'].mean()
train_data['score'].fillna(mean_value, inplace=True)
mean_value = test_data['score'].mean()
test_data['score'].fillna(mean_value, inplace=True)

# Training the Isolation Forest model
X1_train = train_data[features]

model1 = IsolationForest(random_state=42)
model1.fit(X1_train)



# Predicting on the test data
X1_test = test_data[features]
y1_pred = model1.predict(X1_test)


# Saving the results to a file
with open('ResultsIsoForest1.txt', 'w') as file:
    for result in y1_pred:
        file.write(str(result) + '\n')

