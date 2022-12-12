import numpy as np
from numba import jit, njit, prange
from pytictoc import TicToc


def sum_elem(arr: np.ndarray, elem: int) -> None:
    for i in range(len(arr)):
        arr[i] += elem


@njit(cache=True)
def nb_sum_elem(arr: np.ndarray, elem: int) -> None:
    for i in range(len(arr)):
        arr[i] += elem


@njit(cache=True, parallel=True)
def pnb_sum_elem(arr: np.ndarray, elem: int) -> None:
    for i in range(len(arr)):
        arr[i] += elem


if __name__ == "__main__":
    t = TicToc()
    elem = 100
    arr_inp = np.ones(100_000_000)

    t.tic()
    sum_elem(arr_inp, elem)
    t.toc()

    t.tic()
    nb_sum_elem(arr_inp, elem)
    t.toc()

    t.tic()
    nb_sum_elem(arr_inp, elem)
    t.toc()
