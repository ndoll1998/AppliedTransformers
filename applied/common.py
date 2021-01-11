import numpy as np

def align_shape(L:list, shape:tuple, fill_value:float =0) -> np.ndarray:
    """ Fill/Cut an arbitary deep nested list L to match the given shape. """
    assert (len(shape) > 0)

    # special case
    if (0 in shape):
        return np.full(shape, fill_value=fill_value)

    # cut off
    L = L[:shape[0]]

    if len(shape) > 1:
        # recursive match shapes
        L = np.stack([align_shape(l, shape[1:], fill_value=fill_value) for l in L], axis=0)
        # fill current dimension
        F = np.full((shape[0] - L.shape[0],) + shape[1:], fill_value=fill_value)
        L = np.concatenate((L, F), axis=0)
        # return
        return L

    # match the shape of a one dimensional array
    return np.asarray(L + [fill_value] * (shape[0] - len(L)))
