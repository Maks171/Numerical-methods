import numpy as np
from numpy.lib.arraysetops import isin
import scipy as sp
import pickle

from typing import Union, List, Tuple, Optional


def diag_dominant_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Macierz A ma być diagonalnie zdominowana, tzn. wyrazy na przekątnej sa wieksze od pozostałych w danej kolumnie i wierszu
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: macierz diagonalnie zdominowana o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m, int) and m > 0:
        A = np.random.randint(0, 100, (m, m))
        b = np.random.randint(0, 9, (m, 1))
        diag = np.sum(A, axis=0) + np.sum(A, axis=1)
        A += np.diag(diag)

        return A, b
    else:

        return None


def is_diag_dominant(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest diagonalnie zdominowana
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(A, np.ndarray):
        if len(A.shape) == 2:
            if A.shape[0] == A.shape[1]:
                return all((2 * np.abs(np.diag(A))) >= sum(np.abs(A), 1))
    return None


def symmetric_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: symetryczną macierz o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m, int) and m > 0:
        A = np.random.randint(0, 100, (m, m))
        b = np.random.randint(0, 9, (m, 1))
        symmetric_matrix = (A + np.transpose(A))
        return symmetric_matrix, b
    return None


def is_symmetric(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest symetryczna
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(A, np.ndarray):
        if len(A.shape) == 2:
            if A.shape[0] == A.shape[1]:
                return np.allclose(A, np.transpose(A), 1e-05, 1e-05)
    return None


def solve_jacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray,
                 epsilon: Optional[float] = 1e-8, maxiter: Optional[int] = 100) -> Tuple[np.ndarray, int]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych
    Parameters:
    A np.ndarray: macierz współczynników
    b np.ndarray: wektor wartości prawej strony układu
    x_init np.ndarray: rozwiązanie początkowe
    epsilon Optional[float]: zadana dokładność
    maxiter Optional[int]: ograniczenie iteracji
    
    Returns:
    np.ndarray: przybliżone rozwiązanie (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    int: iteracja
    """
    if isinstance(A, np.ndarray) and isinstance(b, np.ndarray) and isinstance(x_init, np.ndarray) and isinstance(epsilon, float) and isinstance(maxiter, int):
        if maxiter > 0 and epsilon > 0 and A.shape[0] == A.shape[1] and b.shape[0] == A.shape[1] and b.shape[0] == x_init.shape[0]:
            D = np.diag(np.diag(A))
            LU = A - D
            x = x_init
            D_inv = np.diag(1 / np.diag(D))
            for i in range(maxiter):
                x_new = np.dot(D_inv, b - np.dot(LU, x))
                if np.linalg.norm(x_new - x) < epsilon:
                    return x_new, i
                x = x_new
            return x, maxiter


def random_matrix_Ab(m:int):
    """Funkcja tworząca zestaw składający się z macierzy A (m,m) i wektora b (m,)  zawierających losowe wartości
    Parameters:
    m(int): rozmiar macierzy
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,m) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m, int) and m > 0:
        matrix = np.random.randint(0, 100, (m, m))
        vector = np.random.randint(0, 100, (m,))
        return matrix, vector
    else:
        return None


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> Optional[float]:
    """
    Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

    Parameters:
        A: macierz A (m,n) zawierająca współczynniki równania
        x: wektor x (n,) zawierający rozwiązania równania
        b: wektor b (m,) zawierający współczynniki po prawej stronie równania
    Results:
        (float)- wartość normy residuom dla podanych parametrów
    """
    return np.linalg.norm(b - A @ x)
