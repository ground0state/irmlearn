from irmlearn import PoissonIRM
import numpy as np
from scipy import stats


def prepare_data():
    np.random.seed(0)
    h_gamma = [
        (40, 10),
        (2, 20),
        (17, 4),
        (20, 1),
    ]

    thetas = []
    for h in h_gamma:
        theta = np.random.gamma(*h, (10, 10))
        thetas.append(theta)

    temp = []
    for i in range(0, 4, 2):
        temp.append(
            np.hstack([thetas[i], thetas[i + 1]])
        )
    theta_ = np.vstack(temp)

    X = stats.poisson.rvs(mu=theta_, size=theta_.shape)
    X_ = X.copy()
    row_index = X.shape[0]
    col_index = X.shape[1]

    X = X[np.random.permutation(row_index)]
    X = X[:, np.random.permutation(col_index)]

    return X, X_


def test_irm():
    X, X_ = prepare_data()

    alpha = 0.5
    a = 5
    b = 5
    max_iter = 10

    model = PoissonIRM(alpha, a, b, max_iter, verbose=False, use_best_iter=True)

    model.fit(X)

    assert np.all(model.feature_labels_ == np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
                                                    dtype=np.int32))
    assert np.all(model.sample_labels_ == np.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                                                   dtype=np.int32))

    assert len(model.n_sample_labels_) == 2
    assert len(model.n_feature_labels_) == 2

    assert X.shape[0] == len(model.sample_labels_)
    assert X.shape[1] == len(model.feature_labels_)

    assert model.sample_labels_.ndim == 1
    assert model.feature_labels_.ndim == 1

    assert isinstance(model.n_sample_labels_, list)
    assert isinstance(model.n_feature_labels_, list)
