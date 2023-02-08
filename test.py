import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

"""снятие ограничений на вывод данных в консоль"""
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

clf_rotate = MLPClassifier(solver='lbfgs',
                           alpha=1e-5,
                           hidden_layer_sizes=(10, 8),
                           random_state=1,
                           # activation="tanh",
                           max_iter=1000000)

clf_move = MLPClassifier(solver='lbfgs',
                         alpha=1e-5,
                         hidden_layer_sizes=(10, 8),
                         random_state=1,
                         # activation="tanh",
                         max_iter=1000000)

clf_reload_pass_fire = MLPClassifier(solver='lbfgs',
                                     alpha=1e-5,
                                     hidden_layer_sizes=(10, 8),
                                     random_state=1,
                                     # activation="tanh",
                                     max_iter=1000000)

train_dataset = pd.read_csv(r"dataset.csv", sep=";", index_col=False)

y_rotate = train_dataset.drop(
    ["% hp"], axis=1).drop(
    ["% patron"], axis=1).drop(
    ["% all patron"], axis=1).drop(
    ["% dist"], axis=1).drop(
    ["look_at_enemy"], axis=1).drop(
    ["look_at_friend"], axis=1).drop(
    ["enemy_on_left"], axis=1).drop(
    ["move"], axis=1).drop(
    ["reload/pass/fire"], axis=1).drop(
    ["enemy_on_right"], axis=1)

y_move = train_dataset.drop(
    ["% hp"], axis=1).drop(
    ["% patron"], axis=1).drop(
    ["% all patron"], axis=1).drop(
    ["% dist"], axis=1).drop(
    ["look_at_enemy"], axis=1).drop(
    ["look_at_friend"], axis=1).drop(
    ["enemy_on_left"], axis=1).drop(
    ["rotate"], axis=1).drop(
    ["reload/pass/fire"], axis=1).drop(
    ["enemy_on_right"], axis=1)

y_reload_pass_fire = train_dataset.drop(
    ["% hp"], axis=1).drop(
    ["% patron"], axis=1).drop(
    ["% all patron"], axis=1).drop(
    ["% dist"], axis=1).drop(
    ["look_at_enemy"], axis=1).drop(
    ["look_at_friend"], axis=1).drop(
    ["enemy_on_left"], axis=1).drop(
    ["move"], axis=1).drop(
    ["rotate"], axis=1).drop(
    ["enemy_on_right"], axis=1)

X = train_dataset.drop(["rotate"], axis=1).drop(["move"], axis=1).drop(["reload/pass/fire"], axis=1)

X_train_rotate, X_test_rotate, y_train_rotate, y_test_rotate = \
    train_test_split(X, y_rotate, test_size=0.2, random_state=42)
X_train_move, X_test_move, y_train_move, y_test_move = \
    train_test_split(X, y_move, test_size=0.2, random_state=42)
X_train_reload_pass_fire, X_test_reload_pass_fire, y_train_reload_pass_fire, y_test_reload_pass_fire = \
    train_test_split(X, y_reload_pass_fire, test_size=0.2, random_state=42)

clf_rotate.fit(X_train_rotate, y_train_rotate.values.ravel())
clf_move.fit(X_train_move, y_train_move.values.ravel())
clf_reload_pass_fire.fit(X_train_reload_pass_fire, y_train_reload_pass_fire.values.ravel())

predictions_rotate = clf_rotate.predict([[0, 0, 0, 0, 0, 0, 0, 0]])
predictions_move = clf_move.predict(X_test_move)
predictions_fire = clf_reload_pass_fire.predict(X_test_reload_pass_fire)

# print(predictions_rotate)
# print(predictions_move)
# print(predictions_fire)

# print("accuracy score", metrics.accuracy_score(y_test_rotate, predictions_rotate))
# print("accuracy score", metrics.accuracy_score(y_test_move, predictions_move))
# print("accuracy score", metrics.accuracy_score(y_test_reload_pass_fire, predictions_fire))

# print(clf_rotate.coefs_)
# print(clf_rotate.n_layers_)
# print(clf_rotate.n_outputs_)
# print(clf_move.coefs_)
# print(clf_reload_pass_fire.coefs_)
