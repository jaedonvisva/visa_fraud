import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

models= ["KNN", "Linear", "Decision Tree", "Random Forest"]
scores = []

df = pd.read_csv('data.csv')

#data pre processing
df = df.drop('ID', axis=1)

x = np.array(df[["X1", "X3",  "X6",  "X7",  "X8",  "X9",  "X10",  "X11", "X12", "X13", "X14", "X15", "X16",  "X17", "X18", "X19", "X20", "X21", "X22", "X23"]])
y = np.array(df["Y"])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# KNN model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=600)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
scores.append(accuracy)
print("KNN Model Score")
print("Accuracy of KNN Model: {0}%".format(accuracy))
print("\n")

#linear regression model
model = linear_model.LinearRegression()
model.fit(x, y)

r_sq = model.score(x, y)
scores.append(r_sq)
print("Linear Model Info and Score:")
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz

model = DecisionTreeClassifier(criterion="entropy" ,max_depth=7,random_state=20)
model.fit(X_train, y_train)
export_graphviz(
        model,
        out_file="tree.dot",
        rounded=True,
        filled=True
    )
scores.append(model.score(X_test, y_test))
print("\n")
print("Decision Tree Classifier Score: {0}".format(model.score(X_test, y_test)))

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train,y_train)


accuracy = model.score(X_test, y_test)
scores.append(accuracy)
print("\n")
print("Random Forest Classifier Model Score")
print("Accuracy of Random Forest Classifier Model: {0}%".format(accuracy))
print("\n")


best_classifier = models[scores.index(max(scores))]
print("Best Classifier: {0}, with an accuracy of {1}%".format(best_classifier, (max(scores) * 100)))