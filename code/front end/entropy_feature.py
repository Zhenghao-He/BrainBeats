import numpy as np
from pathos.multiprocessing import ThreadPool as Pool
import math
from math import factorial
import EntropyHub as eh
import preprocess as pre


# use Richman-Moorman method to get AE(Approximate Entropy) in time domain
def AE(s: list | np.ndarray, r: float = 0.2, m: int = 2):
    s = np.squeeze(s)
    th = r * np.std(s)

    n = len(s)
    # generate matrix like:np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    x = s[np.arange(n - m + 1).reshape(-1, 1) + np.arange(m)]

    ci = lambda xi: ((np.abs(x - xi).max(1) <= th).sum()) / (n - m + 1)  # an anonymous function
    c = Pool().map(ci, x)
    return np.sum(np.log(c)) / (n - m + 1)


# get SE(Sample Entropy) in time domain
def SE(s, r=0.2, m=2):
    list_len = len(s)  # 总长度
    th = r * np.std(s)  # 容限阈值

    def Phi(k):
        list_split = [s[i:i + k] for i in range(0, list_len - k + (k - m))]  # split the list into many sub-lists
        # and notice that the number of sub-lists do NOT change when k=m and when k=m+1!

        Bm = 0.0
        for i in range(0, len(list_split)):  # 遍历每个子向量
            Bm += ((np.abs(list_split[i] - list_split).max(1) <= th).sum() - 1) / (len(list_split) - 1)  # 注意分子和分母都要减1
        return Bm

    ## 多线程
    # x = Pool().map(Phi, [m,m+1])
    # H = - math.log(x[1] / x[0])
    H = - math.log(Phi(m + 1) / Phi(m))
    return H


# get FE(Fuzzy Entropy) in time domain
def FE(s: np.ndarray, r=0.2, m=2, n=2):
    th = r * np.std(s)
    return eh.FuzzEn(s, 2, r=(th, n))[0][-1]


# get PE(permutation entropy) in time domain
def PE(s, m=2, delay=1, normalize=False):
    x = np.array(s)
    hashmult = np.power(m, np.arange(m))

    # _embed is used to generate the new matrix
    def _embed(x, m=2, delay=1):
        N = len(x)
        Y = np.empty((m, N - (m - 1) * delay))
        for i in range(m):
            Y[i] = x[i * delay:i * delay + Y.shape[1]]
        return Y.T

    # argsort is used to sort the elements in each row
    # eg. [9,10,6] the index of 9 is 0, the index of 6 is 2...., and 6 is min
    # so we should get the 1st element to be the index of 6 which is 2, finally we get [201]
    sorted_idx = _embed(x, m, delay=delay).argsort(kind='quicksort')

    # multiply elements in same position
    # hashmult is [1,2]. Like applying a weight to hashmult. And calculate the sum of each row.
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)

    # Return the counts
    _, c = np.unique(hashval, return_counts=True)

    p = np.true_divide(c, c.sum())  # [0.4 0.2 0.4]  2/5=0.4

    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:  # if PE needs to be normalized
        pe /= np.log2(factorial(m))
    return pe
