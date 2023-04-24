import numpy as np
import scipy
import pickle

from typing import Union, List, Tuple


def absolut_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu bezwzględnego. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu bezwzględnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(v, (int, float, list, np.ndarray)) and isinstance(v_aprox, (int, float, list, np.ndarray)):
        if isinstance(v, (int, float)) and isinstance(v_aprox, list):
            lst = np.zeros(len(v_aprox))
            for i in range(len(v_aprox)):
                lst[i] = np.abs(v - v_aprox[i])
            return lst
            
        elif isinstance(v, list) and isinstance(v_aprox, list):
            if len(v) == len(v_aprox):
                lst = np.zeros(len(v_aprox))
                for i in range(len(v)):
                    lst[i] = np.abs(v[i] - v_aprox[i])
                return lst
            else:
                return np.NaN
                
        elif isinstance(v, np.ndarray) and isinstance(v_aprox, np.ndarray):
            if all((m == n) or (m == 1) or (n == 1) for m, n in zip(v.shape[::-1], v_aprox.shape[::-1])):
                return np.abs(v - v_aprox)
            else:
                return np.NaN

        else:
            return np.abs(v - v_aprox)
    else:
        return np.NaN


def relative_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu względnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu względnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(v, (int, float, list, np.ndarray)) and isinstance(v_aprox, (int, float, list, np.ndarray)):
        numerator = absolut_error(v, v_aprox)
        if numerator is np.NaN or (isinstance(v, (int, float)) and v == 0):
            return np.NaN
    
        elif isinstance(v, np.ndarray):
            return np.divide(numerator, v)

        elif isinstance(numerator, np.ndarray) and isinstance(v, list):
            lst = np.zeros(len(v))
            for i in range(len(v)):
                if v[i] == 0:
                    return np.NaN
                lst[i] = numerator[i] / v[i]
            return lst

        else:
            return numerator / v
    else:
        return np.NaN


def p_diff(n: int, c: float) -> float:
    """Funkcja wylicza wartości wyrażeń P1 i P2 w zależności od n i c.
    Następnie zwraca wartość bezwzględną z ich różnicy.
    Szczegóły w Zadaniu 2.
    
    Parameters:
    n Union[int]: 
    c Union[int, float]: 
    
    Returns:
    diff float: różnica P1-P2
                NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(n, int) and isinstance(c, (int, float)):
        b = 2 ** n
        P1 = b - b + c
        P2 = b + c - b
        return np.abs(P1 - P2)
    else:
        return np.NaN


def exponential(x: Union[int, float], n: int) -> float:
    """Funkcja znajdująca przybliżenie funkcji exp(x).
    Do obliczania silni można użyć funkcji scipy.math.factorial(x)
    Szczegóły w Zadaniu 3.
    
    Parameters:
    x Union[int, float]: wykładnik funkcji ekspotencjalnej 
    n Union[int]: liczba wyrazów w ciągu
    
    Returns:
    exp_aprox float: aproksymowana wartość funkcji,
                     NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(x, (int, float)) and isinstance(n, int):
        if n > 0:
            e = 0
            for i in range(n):
                e += x ** i / scipy.math.factorial(i) 
            return e
        else:
            return np.NaN
    else:
        return np.NaN


def coskx1(k: int, x: Union[int, float]) -> float:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 1.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx float: aproksymowana wartość funkcji,
                 NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(k, int) and isinstance(x, (int, float)):
        if k < 0:
            return np.NaN
        elif k == 0:
            return 1
        elif k == 1:
            return np.cos(x)
        else:
            return 2 * np.cos(x) * coskx1(k - 1, x) - coskx1(k - 2, x)
    else:
        return np.NaN


def coskx2(k: int, x: Union[int, float]) -> Tuple[float, float]:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 2.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx, sinkx float: aproksymowana wartość funkcji,
                        NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(k, int) and isinstance(x, (int, float)):
        if k < 0:
            return np.NaN
        elif k == 0:
            return 1, 0
        elif k == 1:
            return np.cos(x), np.sin(x)
        else:
            return np.cos(x) * coskx2(k - 1, x)[0] - np.sin(x) * coskx2(k - 1, x)[1], np.sin(x) * coskx2(k - 1, x)[0] + np.cos(x) * coskx2(k -1, x)[1]
    else:
        return np.NaN



def pi(n: int) -> float:
    """Funkcja znajdująca przybliżenie wartości stałej pi.
    Szczegóły w Zadaniu 5.
    
    Parameters:
    n Union[int, List[int], np.ndarray[int]]: liczba wyrazów w ciągu
    
    Returns:
    pi_aprox float: przybliżenie stałej pi,
                    NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(n, int):
        if n > 0:
            sum = 0
            for i in range(1, n + 1):
                sum += 1 / i ** 2
            return np.sqrt(6 * sum)
        else:
            return np.NaN
    else: 
        return np.NaN