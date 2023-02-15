import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay

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

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for dataset_counter, dataset in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(3, 2, i)
    if dataset_counter == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    i += 1

    ax = plt.subplot(3, 2, i)

    gauss = make_pipeline(StandardScaler(), gauss)
    gauss.fit(X_train, y_train)
    score = gauss.score(X_test, y_test)
    DecisionBoundaryDisplay.from_estimator(
        gauss, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
    )
    print(gauss.predict([[2, 0]]))

    # # Plot the training points
    # ax.scatter(
    #     X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
    # )
    # # Plot the testing points
    # ax.scatter(
    #     X_test[:, 0],
    #     X_test[:, 1],
    #     c=y_test,
    #     cmap=cm_bright,
    #     edgecolors="k",
    #     alpha=0.6,
    # )

    i += 1

plt.tight_layout()
plt.show()
