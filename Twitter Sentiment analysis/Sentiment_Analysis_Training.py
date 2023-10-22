import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import time

start_time = time.time()

data=pd.read_csv('Twitter Sentiment analysis/Features_Target.csv')

Y=data['Target']
X=data.drop('Target',axis=1)
X=data.drop('Weekday',axis=1)
X=data.drop('Month',axis=1)
X=data.drop('Day',axis=1)
X=data.drop('Time',axis=1)

X_train , X_test , Y_train , Y_test = train_test_split(
    X , Y , test_size=0.2 , random_state=100 , stratify = Y
    )

model1=DecisionTreeClassifier(random_state=100)

param_grid = {'max_depth': [30, 40],
              'min_samples_split': [2, 3],
              'min_samples_leaf': [2, 3],
              'max_features': ['sqrt', 'log2']}

grid_search = GridSearchCV(
    estimator=model1, param_grid=param_grid, cv=5, scoring='accuracy'
    )

grid_search.fit(X_train, Y_train)

print("Best hyperparameters for Decision tree:", grid_search.best_params_)
best_model = grid_search.best_estimator_

Y_pred = best_model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print(f"Decision Tree Accuracy: {accuracy}")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")