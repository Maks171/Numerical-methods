import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import isin
import scipy.linalg
from numpy.core._multiarray_umath import ndarray
from numpy.polynomial import polynomial as P
import pickle

# zad1
def polly_A(x: np.ndarray):
    """Funkcja wyznaczajaca współczynniki wielomianu przy znanym wektorze pierwiastków.
    Parameters:
    x: wektor pierwiastków
    Results:
    (np.ndarray): wektor współczynników
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(x, np.ndarray):
        return P.polyfromroots(x)
    else:
        return None

def roots_20(a: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray): wektor współczynników i miejsc zerowych w danej pętli
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(a, np.ndarray):
        a = np.array(a, dtype = float)
        for i in range(len(a)):
            random_sample = np.random.random_sample(a.shape[0]) * 1e-10
            a += random_sample
            return a, P.polyroots(a)
    return None


# zad 2

def frob_a(wsp: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray, np.ndarray, np. ndarray,): macierz Frobenusa o rozmiarze nxn, gdzie n-1 stopień wielomianu,
    wektor własności własnych, wektor wartości z rozkładu schura, wektor miejsc zerowych otrzymanych za pomocą funkcji polyroots

                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(wsp, np.ndarray):
        m = np.eye(wsp.shape[0] - 1)
        zero_vert = np.zeros((wsp.shape[0] - 1, 1))
        m = np.concatenate((zero_vert, m), axis=1)
        m = np.concatenate((m, np.reshape(-wsp, (1, wsp.shape[0]))), axis=0)
        return m, np.linalg.eigvals(m), scipy.linalg.schur(m), P.polyroots(wsp)
    return None



