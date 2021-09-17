import copy

import numpy as np
from scipy.special import betaln, gammaln
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from abc import ABCMeta, abstractmethod

from .utils import crp, log_ewens_sampling_formula, pick_discrete

__all__ = ['IRM', 'PoissonIRM']

MINUS_INFINITY = -float('inf')
DEBUG = False


class IRMBase(BaseEstimator, metaclass=ABCMeta):
    """Infinite Relational Model.
    This is an abstract class of infinite relational model.
    You must implement abstract method.

    Abstract Method
    ----------
    _calc_posterior
        Calculates log-likelihood and returns.

    _draw_sample_label
        Select a sample label and returns.
        If new cluster was selected, returns `-1`.

    _draw_feature_label
        Select a feature label and returns.
        If new cluster was selected, returns `-1`.

    Refs
    ----------
    Charles Kemp, Joshua B. Tenenbaum, Thomas L. Griffiths, Takeshi Yamada, and Naonori Ueda. 2006.
    Learning systems of concepts with an infinite relational model.
    In Proceedings of the 21st national conference on Artificial intelligence - Volume 1 (AAAI'06).
    AAAI Press, 381–388.

    Parameters
    ----------
    alpha : float
        concentration parameter.

    max_iter : int, default=500
        Maximum number of iterations of the IRM algorithm to run.

    max_n_sample_labels : int, default=100
        Maximum number of labels of the IRM algorithm to run.
        Early stopping with iterations that exceed the max_n_labels.
        Initialization does not stop fitting, but iterates at least once.

    max_n_feature_labels : int, default=100
        Maximum number of labels of the IRM algorithm to run.
        Early stopping with iterations that exceed the max_n_labels.
        Initialization does not stop fitting, but iterates at least once.

    use_best_iter : bool, default=True
        Use the result when the likelihood was maximum during the fitting.

    patience : int, default=None
        Number of consecutive non improving epoch before early stopping.

    verbose : bool, default=True
        Controls the verbosity: the higher, the more messages.

        - True : n_sample_labels and n_feature_labels are displayed when updated;
        - False : nothing is displayed;

    random_state : int, default=0
        Controls the randomness of infinite relational model.
    """

    def __init__(self,
                 alpha,
                 max_iter=100,
                 max_n_sample_labels=100,
                 max_n_feature_labels=100,
                 use_best_iter=True,
                 patience=None,
                 verbose=True,
                 random_state=0):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.max_n_sample_labels = int(max_n_sample_labels)
        self.max_n_feature_labels = int(max_n_feature_labels)
        self.use_best_iter = bool(use_best_iter)
        self.patience = patience
        self.verbose = bool(verbose)
        self.random_state = int(random_state)

        self.sample_labels_ = None
        self.n_sample_labels_ = None
        self.feature_labels_ = None
        self.n_feature_labels_ = None

        self._logv_max = MINUS_INFINITY
        self._logv_cur = None
        self._step = 0
        self._cur_iter = None

    def _check_params(self, X):
        # alpha
        if self.alpha <= 0:
            raise ValueError(
                f"alpha must be > 0, got {self.alpha} instead.")
        # max_iter
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter must be > 0, got {self.max_iter} instead.")
        # max_n_sample_labels
        if self.max_n_sample_labels <= 0:
            raise ValueError(
                f"max_n_sample_labels must be > 0, got {self.max_n_sample_labels} instead.")
        # max_n_feature_labels
        if self.max_n_feature_labels <= 0:
            raise ValueError(
                f"max_n_feature_labels must be > 0, got {self.max_n_feature_labels} instead.")
        # patience
        if self.patience is not None:
            self.patience = int(self.patience)
            if self.patience <= 0:
                raise ValueError(
                    f"patience must be > 0, got {self.patience} instead.")
        # verbose
        if not isinstance(self.verbose, bool):
            raise ValueError(
                f"verbose must be bool, got {self.verbose} instead.")
        # random_state
        if not isinstance(self.random_state, int):
            raise ValueError(
                f"random_state must be int, got {self.random_state} instead.")
        if self.random_state < 0:
            raise ValueError(
                f"random_state must be >= 0, got {self.random_state} instead.")

    def _initialize(self, X):
        n_samples, n_features = X.shape
        np.random.seed(self.random_state)
        self.sample_labels_, self.n_sample_labels_ = crp(n_samples, self.alpha)
        self.feature_labels_, self.n_feature_labels_ = crp(
            n_features, self.alpha)
        self.sample_labels_ = np.array(self.sample_labels_, dtype=np.int32)
        self.feature_labels_ = np.array(self.feature_labels_, dtype=np.int32)

        self._sample_labels_best = self.sample_labels_.copy()
        self._n_sample_labels_best = copy.deepcopy(self.n_sample_labels_)
        self._feature_labels_best = self.feature_labels_.copy()
        self._n_feature_labels_best = copy.deepcopy(self.n_feature_labels_)

        self._best_iter = 0
        self.history_ = []

    def fit(self, X, y=None):
        """
        Perform clustering on `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : Ignored
            Not used, present for API consistency by convention.
        """
        # validation
        X = check_array(X)
        self._check_params(X)
        self._initialize(X)

        for i in range(self.max_iter):
            self._cur_iter = i + 1
            if self._update(X):
                break
        self._commit_best_update()

        if self.verbose:
            print('=' * 10, 'Finished!', '=' * 10, flush=True)
            print(
                f"best_iter={self._best_iter} -- sample_n_labels: {len(self.n_sample_labels_)}" +
                f" -- feature_n_labels: {len(self.n_feature_labels_)}", flush=True
            )
        return self

    def _commit_best_update(self):
        if self.use_best_iter:
            self.sample_labels_ = self._sample_labels_best.copy()
            self.n_sample_labels_ = copy.deepcopy(self._n_sample_labels_best)
            self.feature_labels_ = self._feature_labels_best.copy()
            self.n_feature_labels_ = copy.deepcopy(self._n_feature_labels_best)

    def _update(self, X):
        n_samples, n_features = X.shape
        sample_iter = iter(range(n_samples))
        feature_iter = iter(range(n_features))

        for idx in np.random.permutation([0] * n_samples + [1] * n_features):
            if idx == 0:
                k = next(sample_iter)
                self._del_sample_label_k(k)
                self._update_sample_label_k(X, k)
            else:
                l = next(feature_iter)
                self._del_feature_label_l(l)
                self._update_feature_label_l(X, l)

        self._calc_posterior(X)
        return self._end_update()

    def _end_update(self):
        is_early_stop = False
        self.history_.append(self._logv_cur)

        if self._logv_cur > self._logv_max:
            self._logv_max = self._logv_cur
            self._best_iter = self._cur_iter

            self._sample_labels_best = self.sample_labels_.copy()
            self._n_sample_labels_best = copy.deepcopy(self.n_sample_labels_)
            self._feature_labels_best = self.feature_labels_.copy()
            self._n_feature_labels_best = copy.deepcopy(self.n_feature_labels_)

            self._step = 0
        else:
            self._step += 1
            if (self.patience is not None) and (self._step > self.patience):
                is_early_stop = True
                if self.verbose:
                    print(f'iter={self._cur_iter} -- Reached patience', flush=True)

        if len(self.n_sample_labels_) >= self.max_n_sample_labels:
            is_early_stop = True
            if self.verbose:
                print(f'iter={self._cur_iter} -- Reached max_n_sample_labels', flush=True)

        if len(self.n_feature_labels_) >= self.max_n_feature_labels:
            is_early_stop = True
            if self.verbose:
                print(f'iter={self._cur_iter} -- Reached max_n_feature_labels', flush=True)
        return is_early_stop

    def _del_sample_label_k(self, k):
        """Note that this is a destructive method.
        """
        label = self.sample_labels_[k]
        self.n_sample_labels_[label] = self.n_sample_labels_[label] - 1
        if self.n_sample_labels_[label] == 0:
            del self.n_sample_labels_[label]
            if self.verbose:
                print(f"iter={self._cur_iter} -- sample label deleted, n_sample_labels: {len(self.n_sample_labels_)}", flush=True)
            # Need to decrement label numbers for labels greater than the one
            # deleted...
            self.sample_labels_[self.sample_labels_ >= label] = \
                self.sample_labels_[self.sample_labels_ >= label] - 1

    def _del_feature_label_l(self, l):
        """Note that this is a destructive method.
        """
        label = self.feature_labels_[l]
        self.n_feature_labels_[label] = self.n_feature_labels_[label] - 1
        if self.n_feature_labels_[label] == 0:
            del self.n_feature_labels_[label]
            if self.verbose:
                print(f"iter={self._cur_iter} -- feature label deleted, n_feature_labels: {len(self.n_feature_labels_)}", flush=True)
            # Need to decrement label numbers for labels greater than the one
            # deleted...
            self.feature_labels_[self.feature_labels_ >= label] =\
                self.feature_labels_[self.feature_labels_ >= label] - 1

    def _update_sample_label_k(self, X, k):
        label = self._draw_sample_label(X, k)

        # If we selected to create a new cluster, then draw parameters for that cluster.
        if label == -1:
            self.n_sample_labels_.append(1)
            self.sample_labels_[k] = len(self.n_sample_labels_) - 1
            if self.verbose:
                print(
                    f"iter={self._cur_iter} -- New sample label created, n_sample_labels: {len(self.n_sample_labels_)}", flush=True)
        else:  # Otherwise just increment the count for the cloned cluster.
            self.sample_labels_[k] = label
            self.n_sample_labels_[label] = self.n_sample_labels_[label] + 1

    def _update_feature_label_l(self, X, l):
        label = self._draw_feature_label(X, l)

        # If we selected to create a new cluster, then draw parameters for that cluster.
        if label == -1:
            self.n_feature_labels_.append(1)
            self.feature_labels_[l] = len(self.n_feature_labels_) - 1
            if self.verbose:
                print(
                    f"iter={self._cur_iter} -- New feature label created, n_feature_labels: {len(self.n_feature_labels_)}", flush=True)
        else:  # Otherwise just increment the count for the cloned cluster.
            self.feature_labels_[l] = label
            self.n_feature_labels_[label] = self.n_feature_labels_[label] + 1

    @abstractmethod
    def _calc_posterior(self, X):
        pass

    @abstractmethod
    def _draw_sample_label(self, X, k):
        pass

    @abstractmethod
    def _draw_feature_label(self, X, l):
        pass


class IRM(IRMBase):
    """Infinite Relational Model.
    This is an infinite relational model for 0/1 data.

    Parameters
    ----------
    alpha : float
        concentration parameter.

    a : float
        shape parameter of beta distribution, which is a prior conjugate distribution.
        It is estimated that the larger this parameter is, the more likely it is to be 1.

    b : float
        shape parameter of beta distribution, which is a prior conjugate distribution.
        It is estimated that the larger this parameter is, the more likely it is to be 0.

    max_iter : int, default=500
        Maximum number of iterations of the IRM algorithm to run.

    max_n_sample_labels : int, default=100
        Maximum number of labels of the IRM algorithm to run.
        Early stopping with iterations that exceed the max_n_labels.
        Initialization does not stop fitting, but iterates at least once.

    max_n_feature_labels : int, default=100
        Maximum number of labels of the IRM algorithm to run.
        Early stopping with iterations that exceed the max_n_labels.
        Initialization does not stop fitting, but iterates at least once.

    use_best_iter : bool, default=True
        Use the result when the likelihood was maximum during the fitting.

    patience : int, default=None
        Number of consecutive non improving epoch before early stopping.

    verbose : bool, default=True
        Controls the verbosity: the higher, the more messages.

        - True : n_sample_labels and n_feature_labels are displayed when updated;
        - False : nothing is displayed;

    random_state : int, default=0
        Controls the randomness of infinite relational model.
    """

    def __init__(self,
                 alpha,
                 a,
                 b,
                 max_iter=500,
                 max_n_sample_labels=100,
                 max_n_feature_labels=100,
                 use_best_iter=True,
                 patience=None,
                 verbose=True):
        super().__init__(
            alpha=alpha,
            max_iter=max_iter,
            max_n_sample_labels=max_n_sample_labels,
            max_n_feature_labels=max_n_feature_labels,
            use_best_iter=use_best_iter,
            patience=patience,
            verbose=verbose)
        self.a = a
        self.b = b

    def _check_params(self, X):
        super()._check_params(X)
        # a
        if self.a <= 0:
            raise ValueError(
                f"a must be > 0, got {self.a} instead.")

    def _calc_posterior(self, X):
        logv = log_ewens_sampling_formula(self.alpha, self.n_sample_labels_) - gammaln(len(self.n_sample_labels_) + 1)
        logv += log_ewens_sampling_formula(self.alpha, self.n_feature_labels_) - gammaln(len(self.n_feature_labels_) + 1)
        for i in range(len(self.n_sample_labels_)):
            for j in range(len(self.n_feature_labels_)):
                X_ = X[self.sample_labels_ == i, :][:, self.feature_labels_ == j].copy()
                n_pp = X_.sum()
                nbar_pp = (1 - X_).sum()
                logv += betaln(n_pp + self.a, nbar_pp + self.b) - betaln(self.a, self.b)
        self._logv_cur = logv

    def _draw_sample_label(self, X, k):
        logps = []
        logp = np.log(self.alpha)
        for j in range(len(self.n_feature_labels_)):
            n_kpp = X[k, self.feature_labels_ == j].sum()
            nbar_kpp = (1 - X[k, self.feature_labels_ == j]).sum()
            logp += betaln(n_kpp + self.a, nbar_kpp + self.b) - betaln(self.a, self.b)
        logps.append(logp)

        for i in range(len(self.n_sample_labels_)):
            self.sample_labels_[k] = i
            logp = np.log(self.n_sample_labels_[i])
            for j in range(len(self.n_feature_labels_)):
                X_ = X[self.sample_labels_ == i, :][:, self.feature_labels_ == j].copy()
                n_pp = X_.sum()
                nbar_pp = (1 - X_).sum()
                n_kp = X[k, self.feature_labels_ == j].sum()
                nbar_kp = (1 - X[k, self.feature_labels_ == j]).sum()
                n_mkp = n_pp - n_kp
                nbar_mkp = nbar_pp - nbar_kp
                logp += betaln(n_pp + self.a, nbar_pp + self.b) - betaln(n_mkp + self.a, nbar_mkp + self.b)
                if DEBUG:
                    print("j", j, "n_kp", n_kp, "n_mkp", n_mkp, "nbar_kp", nbar_kp, "nbar_mkp", nbar_mkp, "logp", logp)
            logps.append(logp)
        logps = np.array(logps)
        if len(logps) >= 2:
            logps -= logps.max()
        ps = np.exp(logps)
        label = pick_discrete(ps) - 1
        return label

    def _draw_feature_label(self, X, l):
        logps = []
        logp = np.log(self.alpha)
        for i in range(len(self.n_sample_labels_)):
            n_plp = X[self.sample_labels_ == i, l].sum()
            nbar_plp = (1 - X[self.sample_labels_ == i, l]).sum()
            logp += betaln(n_plp + self.a, nbar_plp + self.b) - betaln(self.a, self.b)
        logps.append(logp)

        for j in range(len(self.n_feature_labels_)):
            self.feature_labels_[l] = j
            logp = np.log(self.n_feature_labels_[j])
            for i in range(len(self.n_sample_labels_)):
                X_ = X[self.sample_labels_ == i, :][:, self.feature_labels_ == j].copy()
                n_pp = X_.sum()
                nbar_pp = (1 - X_).sum()
                n_pl = X[self.sample_labels_ == i, l].sum()
                nbar_pl = (1 - X[self.sample_labels_ == i, l]).sum()
                n_pml = n_pp - n_pl
                nbar_pml = nbar_pp - nbar_pl
                logp += betaln(n_pp + self.a, nbar_pp + self.b) - betaln(n_pml + self.a, nbar_pml + self.b)
                if DEBUG:
                    print("i", i, "n_pl", n_pl, "n_pml", n_pml, "nbar_pl", nbar_pl, "nbar_pml", nbar_pml, "logp", logp)
            logps.append(logp)
        logps = np.array(logps)
        if len(logps) >= 2:
            logps -= logps.max()
        ps = [np.exp(logp) for logp in logps]
        label = pick_discrete(ps) - 1
        return label


class PoissonIRM(IRMBase):
    """Poisson Infinite Relational Model.
    This is an infinite relational model for count data.

    Parameters
    ----------
    alpha : float
        concentration parameter.

    a : float
        shape parameter of gamma distribution, which is a prior conjugate distribution.
        It is estimated that the larger this parameter is, the larger the count number is.

    b : float
        shape parameter of gamma distribution, which is a prior conjugate distribution.
        It is estimated that the larger this parameter is, the larger the count number is.

    max_iter : int, default=500
        Maximum number of iterations of the IRM algorithm to run.

    max_n_sample_labels : int, default=100
        Maximum number of labels of the IRM algorithm to run.
        Early stopping with iterations that exceed the max_n_labels.
        Initialization does not stop fitting, but iterates at least once.

    max_n_feature_labels : int, default=100
        Maximum number of labels of the IRM algorithm to run.
        Early stopping with iterations that exceed the max_n_labels.
        Initialization does not stop fitting, but iterates at least once.

    use_best_iter : bool, default=True
        Use the result when the likelihood was maximum during the fitting.

    patience : int, default=None
        Number of consecutive non improving epoch before early stopping.

    verbose : bool, default=True
        Controls the verbosity: the higher, the more messages.

        - True : n_sample_labels and n_feature_labels are displayed when updated;
        - False : nothing is displayed;

    random_state : int, default=0
        Controls the randomness of infinite relational model.
    """

    def __init__(self,
                 alpha,
                 a,
                 b,
                 max_iter=500,
                 max_n_sample_labels=100,
                 max_n_feature_labels=100,
                 use_best_iter=True,
                 patience=None,
                 verbose=True):
        super().__init__(
            alpha=alpha,
            max_iter=max_iter,
            max_n_sample_labels=max_n_sample_labels,
            max_n_feature_labels=max_n_feature_labels,
            use_best_iter=use_best_iter,
            patience=patience,
            verbose=verbose)
        self.a = a
        self.b = b

    def _calc_posterior(self, X):
        logv = log_ewens_sampling_formula(self.alpha, self.n_sample_labels_) - gammaln(len(self.n_sample_labels_) + 1)
        logv += log_ewens_sampling_formula(self.alpha, self.n_feature_labels_) - gammaln(len(self.n_feature_labels_) + 1)
        for i in range(len(self.n_sample_labels_)):
            for j in range(len(self.n_feature_labels_)):
                X_ = X[self.sample_labels_ == i, :][:, self.feature_labels_ == j].copy()
                c_pp = X_.sum()
                m_pp = X_.size
                logv += self.log_g(c_pp + self.a, m_pp + self.b) - self.log_g(self.a, self.b) - np.sum(gammaln(X_ + 1))
        self._logv_cur = logv

    def _draw_sample_label(self, X, k):
        logps = []
        logp = np.log(self.alpha)
        for j in range(len(self.n_feature_labels_)):
            c_kpp = X[k, self.feature_labels_ == j].sum()
            m_kpp = X[k, self.feature_labels_ == j].size
            logp += self.log_g(c_kpp + self.a, m_kpp + self.b) - self.log_g(self.a, self.b)
        logps.append(logp)

        for i in range(len(self.n_sample_labels_)):
            self.sample_labels_[k] = i
            logp = np.log(self.n_sample_labels_[i])
            for j in range(len(self.n_feature_labels_)):
                X_ = X[self.sample_labels_ == i, :][:, self.feature_labels_ == j].copy()
                c_pp = X_.sum()
                m_pp = X_.size
                c_kp = X[k, self.feature_labels_ == j].sum()
                m_kp = X[k, self.feature_labels_ == j].size
                c_mkp = c_pp - c_kp
                m_mkp = m_pp - m_kp
                logp += self.log_g(c_pp + self.a, m_pp + self.b) - self.log_g(c_mkp + self.a, m_mkp + self.b)
            logps.append(logp)
        logps = np.array(logps)
        if len(logps) >= 2:
            logps -= logps.max()
        ps = np.exp(logps)
        label = pick_discrete(ps) - 1
        return label

    def _draw_feature_label(self, X, l):
        logps = []
        logp = np.log(self.alpha)
        for i in range(len(self.n_sample_labels_)):
            c_plp = X[self.sample_labels_ == i, l].sum()
            m_plp = X[self.sample_labels_ == i, l].size
            logp += self.log_g(c_plp + self.a, m_plp + self.b) - self.log_g(self.a, self.b)
        logps.append(logp)

        for j in range(len(self.n_feature_labels_)):
            self.feature_labels_[l] = j
            logp = np.log(self.n_feature_labels_[j])
            for i in range(len(self.n_sample_labels_)):
                X_ = X[self.sample_labels_ == i, :][:, self.feature_labels_ == j].copy()
                c_pp = X_.sum()
                m_pp = X_.size
                c_pl = X[self.sample_labels_ == i, l].sum()
                m_pl = X[self.sample_labels_ == i, l].size
                c_pml = c_pp - c_pl
                m_pml = m_pp - m_pl
                logp += self.log_g(c_pp + self.a, m_pp + self.b) - self.log_g(c_pml + self.a, m_pml + self.b)
            logps.append(logp)
        logps = np.array(logps)
        if len(logps) >= 2:
            logps -= logps.max()
        ps = [np.exp(logp) for logp in logps]
        label = pick_discrete(ps) - 1
        return label

    def log_g(self, a, b):
        return gammaln(a) - a * np.log(b)
