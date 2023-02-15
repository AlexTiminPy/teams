import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

gauss = GaussianProcessClassifier(1.0 * RBF(1.0))

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
]

# iterate over datasets
for dataset_counter, dataset in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.01, random_state=42
    )

    gauss = make_pipeline(StandardScaler(), gauss)
    gauss.fit(X_train, y_train)
    score = gauss.score(X_test, y_test)

import pickle

# save
print("start save rotate_model")
with open('modelsSave/gaussModel.pkl', 'wb') as f:
    pickle.dump(gauss, f)
print("saving rotate_model")
