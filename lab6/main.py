import numpy as np
import pickle

from typing import Union, List, Tuple

def random_matrix_Ab(m:int):
    """Funkcja tworząca zestaw składający się z macierzy A (m,m) i wektora b (m,)  zawierających losowe wartości
    Parameters:
    m(int): rozmiar macierzy
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,m) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m, int) and m > 0:
        matrix = np.random.randint(0, 100, size = (m, m))
        vector = np.random.randint(0, 100, size = (m,))
        return matrix, vector
    else:
        return None

def residual_norm(A:np.ndarray,x:np.ndarray, b:np.ndarray):
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

      Parameters:
      A: macierz A (m,m) zawierająca współczynniki równania 
      x: wektor x (m.) zawierający rozwiązania równania 
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania

      Results:
      (float)- wartość normy residuom dla podanych parametrów"""
    if isinstance(A, np.ndarray) and isinstance (x, np.ndarray) and isinstance(b, np.ndarray):
        if x.shape == b.shape and A.shape[1] == A.shape[0]:
            A1 = A @ np.transpose(x)
            res = b - A1
            return np.linalg.norm(res)
        else:
            return None
    else:
        return None

def log_sing_value(n:int, min_order:Union[int,float], max_order:Union[int,float]):
    """Funkcja generująca wektor wartości singularnych rozłożonych w skali logarytmiczne
    
        Parameters:
         n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
         min_order(int,float): rząd najmniejszej wartości w wektorze wartości singularnych
         max_order(int,float): rząd największej wartości w wektorze wartości singularnych
         Results:
         np.ndarray - wektor nierosnących wartości logarytmicznych o wymiarze (n,) zawierający wartości logarytmiczne na zadanym przedziale
         """
    if isinstance(n, int) and isinstance(min_order, (int, float)) and isinstance(max_order, (int, float)):
        if max_order > min_order and n > 0:
            return np.flip(np.logspace(min_order, max_order, n))
        else:
            return None
    else:
        return None
    
def order_sing_value(n:int, order:Union[int,float] = 2, site:str = 'gre'):
    """Funkcja generująca wektor losowych wartości singularnych (n,) będących wartościami zmiennoprzecinkowymi losowanymi przy użyciu funkcji np.random.rand(n)*10. 
        A następnie ustawiająca wartość minimalną (site = 'low') albo maksymalną (site = 'gre') na wartość o  10**order razy mniejszą/większą.
    
        Parameters:
        n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
        order(int,float): rząd przeskalowania wartości skrajnej
        site(str): zmienna wskazująca stronnę zmiany:
            - site = 'low' -> sing_value[-1] * 10**order
            - site = 'gre' -> sing_value[0] * 10**order
        
        Results:
        np.ndarray - wektor wartości singularnych o wymiarze (n,) zawierający wartości logarytmiczne na zadanym przedziale
        """
    if isinstance(n, int) and isinstance(order, (int, float)) and isinstance(site, str):
        if n > 0:
            vector = np.random.rand(n)*10
            if site == "gre":
                max_idx = np.argmax(vector)
                vector[max_idx] += 10 ** order
            elif site == 'low':
                min_idx = np.argmin(vector)
                vector[min_idx] -= 10 ** order
            else:
                return None
            sorted = np.sort(vector)
            return np.flip(sorted)
        else:
            return None    
    else:
        return None

def create_matrix_from_A(A:np.ndarray, sing_value:np.ndarray):
    """Funkcja generująca rozkład SVD dla macierzy A i zwracająca otworzenie macierzy A z wykorzystaniem zdefiniowanego wektora warości singularnych

            Parameters:
            A(np.ndarray): rozmiarz macierzy A (m,m)
            sing_value(np.ndarray): wektor wartości singularnych (m,)


            Results:
            np.ndarray: macierz (m,m) utworzoną na podstawie rozkładu SVD zadanej macierzy A z podmienionym wektorem wartości singularnych na wektor sing_valu """
    if isinstance(A, np.ndarray) and isinstance(sing_value, np.ndarray):
        if A.shape[0] == sing_value.shape[0] and A.shape[1] == A.shape[0]:
            U, _, V = np.linalg.svd(A)
            return np.dot(U * sing_value, V)
        else:
            return None
    else:
        return None
    