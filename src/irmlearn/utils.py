import bisect

import numpy as np


def pick_discrete(p):
    """Pick a discrete integer between 0 and len(p) - 1 with probability given by (normalized) p
    array.  Note that p array will be normalized here."""
    c = np.cumsum(p)
    c = c / c[-1]  # Normalize
    u = np.random.uniform()
    return bisect.bisect(c, u)


def crp(n, alpha):
    """Chinese restaurant process.

    Parameters
    ----------
    n : int
        num of people to seat.
    alpha : float
        Concentration.

    Return
    ----------
    assignments : list
        table_id for each people.
    n_assignments : list
        partition.
    """
    n = int(n)
    alpha = float(alpha)
    assert n >= 1
    assert alpha > 0

    assignments = [0]
    n_assignments = [1]
    for _ in range(2, n + 1):
        table_id = pick_discrete(n_assignments) - 1
        if table_id == -1:
            n_assignments.append(1)
            assignments.append(len(n_assignments) - 1)
        else:
            n_assignments[table_id] = n_assignments[table_id] + 1
            assignments.append(table_id)

    return assignments, n_assignments


def log_ewens_sampling_formula(alpha, count_list):
    n = np.sum(count_list)
    c = len(count_list)
    res = c * np.log(alpha)
    for i in range(c):
        param = count_list[i] - 1
        while param > 0:
            res += np.log(param)
            param -= 1
    for i in range(n):
        res -= np.log(i + alpha)
    return res.item()
