from sklearn.neural_network import MLPClassifier
from sklearn import metrics

clf = MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    hidden_layer_sizes=(6, 3),
                    random_state=1,
                    max_iter=1000000)

x_train_dataset1 = [
    [0, 1, 0, 1, 0, 1],
]

y_train_dataset1 = [
    [0, 0, 0],
]

x_test_dataset1 = [
    [0, 1, 0, 1, 0, 1],
]

y_test_dataset1 = [
    [0, 0, 0],
]

x_train_dataset2 = [
    [1, 0, 1, 0, 1, 0],
]

y_train_dataset2 = [
    [1, 1, 1],
]

x_test_dataset2 = [
    [1, 0, 1, 0, 1, 0],
]

y_test_dataset2 = [
    [1, 1, 1],
]

clf.fit(x_train_dataset1, y_train_dataset1)
clf.fit(x_train_dataset2, y_train_dataset2)

for i in clf.coefs_:
    print(i)

predictions1 = clf.predict(x_test_dataset1)
predictions2 = clf.predict(x_test_dataset2)
print(predictions1)
print(predictions2)

print("accuracy score", metrics.accuracy_score(y_test_dataset1, predictions1))
print("accuracy score", metrics.accuracy_score(y_test_dataset2, predictions2))
