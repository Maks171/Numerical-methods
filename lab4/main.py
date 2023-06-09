##
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt

from typing import Union, List, Tuple

def chebyshev_nodes(n:int=10)-> np.ndarray:
    """Funkcja tworząca wektor zawierający węzły czybyszewa w postaci wektora (n+1,)
    
    Parameters:
    n(int): numer ostaniego węzła Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(n, int) and n>=0:
        arr = np.zeros(n+1)
        for k in range(n+1):
            arr[k] = np.cos(k*np.pi/n)
        return arr
    else:
        return None
    
def bar_czeb_weights(n:int=10)-> np.ndarray:
    """Funkcja tworząca wektor wag dla węzłów czybyszewa w postaci (n+1,)
    
    Parameters:
    n(int): numer ostaniej wagi dla węzłów Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor wag dla węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(n, int) and n>=0:
        arr = np.zeros(n+1)
        for j in range(n+1):
            if j == 0 or j == n:
                dj = 0.5
                arr[j] = (-1)**j * dj
            elif j > 0 and j < n:
                dj = 1
                arr[j] = (-1)**j * dj
            else:
                return None
        return arr
    else:
        return None
    
def  barycentric_inte(xi:np.ndarray,yi:np.ndarray,wi:np.ndarray,x:np.ndarray)-> np.ndarray:
    """Funkcja przprowadza interpolację metodą barycentryczną dla zadanych węzłów xi
        i wartości funkcji interpolowanej yi używając wag wi. Zwraca wyliczone wartości
        funkcji interpolującej dla argumentów x w postaci wektora (n,) gdzie n to dłógość
        wektora n. 
    
    Parameters:
    xi(np.ndarray): węzły interpolacji w postaci wektora (m,), gdzie m > 0
    yi(np.ndarray): wartości funkcji interpolowanej w węzłach w postaci wektora (m,), gdzie m>0
    wi(np.ndarray): wagi interpolacji w postaci wektora (m,), gdzie m>0
    x(np.ndarray): argumenty dla funkcji interpolującej (n,), gdzie n>0 
     
    Results:
    np.ndarray: wektor wartości funkcji interpolujący o rozmiarze (n,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(xi, np.ndarray) and isinstance(yi, np.ndarray) and isinstance(wi, np.ndarray) and isinstance(x, np.ndarray):
        if xi.shape == yi.shape and yi.shape == wi.shape:
            Y = []
            L = wi/(x-xi)
            Y.append(yi @ L / sum(L))
            Y = np.array(Y)
            return Y
    else:
        return None
    
def L_inf(xr:Union[int, float, List, np.ndarray],x:Union[int, float, List, np.ndarray])-> float:
    """Obliczenie normy  L nieskończonośćg. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach biblioteki numpy.
    
    Parameters:
    xr (Union[int, float, List, np.ndarray]): wartość dokładna w postaci wektora (n,)
    x (Union[int, float, List, np.ndarray]): wartość przybliżona w postaci wektora (n,1)
    
    Returns:
    float: wartość normy L nieskończoność,
                                    NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(xr, (int, float)) and isinstance(x, (int, float)):
        return np.abs(xr - x)

    if isinstance(xr, List) and isinstance(x, List):
        return np.abs(max(xr) - max(x))

    if isinstance(xr, np.ndarray) and isinstance(x, np.ndarray):
        if xr.shape == x.shape:
            return max(np.abs(xr - x))
        else:
            return np.NaN
    else:
        return np.NaN   



f1 = lambda x: np.sign(x) * x + x ** 2

f2 = lambda x: np.sign(x) * x ** 2

f3 = lambda x: (np.abs(np.sin(5 * x))) ** 3

f41 = lambda x: 1 / (1 + 1 * x ** 2)
f425 = lambda x: 1 / (1 + 25 * x ** 2)
f4100 = lambda x: 1 / (1 + 100 * x ** 2)

f5 = lambda x: np.sign(x)
