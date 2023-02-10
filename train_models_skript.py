import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

from sklearn.neural_network import MLPClassifier
import pandas as pd

"""снятие ограничений на вывод данных в консоль"""
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

clf_rotate = MLPClassifier(solver='lbfgs',
                           alpha=1e-5,
                           hidden_layer_sizes=(10, 8),
                           random_state=1,
                           max_iter=1000000)

clf_move = MLPClassifier(solver='lbfgs',
                         alpha=1e-5,
                         hidden_layer_sizes=(10, 8),
                         random_state=1,
                         max_iter=1000000)

clf_reload_pass_fire = MLPClassifier(solver='lbfgs',
                                     alpha=1e-5,
                                     hidden_layer_sizes=(10, 8),
                                     random_state=1,
                                     max_iter=1000000)

train_dataset = pd.read_csv(r"dataset.csv", sep=";", index_col=False)

y_rotate = \
    train_dataset.drop(
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

y_move = \
    train_dataset.drop(
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

y_reload_pass_fire = \
    train_dataset.drop(
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

clf_rotate.fit(X, y_rotate.values.ravel())
clf_move.fit(X, y_move.values.ravel())
clf_reload_pass_fire.fit(X, y_reload_pass_fire.values.ravel())

import pickle

# save
print("start save rotate_model")
with open('modelsSave/rotate_model.pkl', 'wb') as f:
    pickle.dump(clf_rotate, f)
print("saving rotate_model")

print("start save move_model")
with open('modelsSave/move_model.pkl', 'wb') as f:
    pickle.dump(clf_move, f)
print("saving move_model")

print("start save reload_pass_fire_model")
with open('modelsSave/reload_pass_fire_model.pkl', 'wb') as f:
    pickle.dump(clf_reload_pass_fire, f)
print("saving reload_pass_fire_model")

# load
# with open('model.pkl', 'rb') as f:
#     clf2 = pickle.load(f)
# clf2.predict(X[0:1])
