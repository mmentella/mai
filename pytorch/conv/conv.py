import numpy as np


def conv1D(x, w, p=0, s=1):
    """
    x : input vector
    w : filter
    p : padding size
    s : stride
    """
    assert len(w) <= len(x), "x should be bigger than w"
    assert p >= 0, "padding cannot be negative"

    w_r = np.array(w[::-1])  # rotation of w
    x_padded = np.array(x)

    if p > 0:
        zeros = np.zeros(shape=p)
        x_padded = np.concatenate(
            [zeros, x_padded, zeros]
        )  # add zeros around original vector

    out = []
    # iterate through the original array s cells per step
    for i in range(0, int((len(x_padded) - len(w_r))) + 1, s):
        out.append(
            np.sum(x_padded[i : i + w_r.shape[0]] * w_r)
        )  # formula we have seen before
    return np.array(out)


x = [
    -11,
    19,
    36,
    55,
    44,
    67,
    76,
    64,
    2,
    17,
    6,
    -11,
    12,
    6,
    18,
    -9,
    52,
    35,
    56,
    1,
    23,
    47,
    56,
    41,
]
w = [-1,1,-1,1,-1]

print(conv1D(x, w, 2, 1))
