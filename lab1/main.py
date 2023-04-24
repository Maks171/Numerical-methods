import main
import numpy as np
import scipy
import math

def cylinder_area(r:float,h:float):
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca
    
    Returns:
    float: pole powierzchni walca 
    """
    if r < 0 or h < 0:
        return np.NaN
    else:
        return 2*math.pi*r*r+2*math.pi*r*h 
    


def fib(n:int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    if isinstance(n, float):
        return None
    elif n == 0:
        return None
    elif n == 1:
        return np.array([1])
    elif n > 1:
        a = 1
        b = 1
        i = 0
        lista = []
        while (i < n):
            lista.append(a)
            suma = a + b
            a = b
            b = suma
            i += 1
        return np.array([lista])

        

def matrix_calculations(a:float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    M = np.array([[a,1,-a],[0,1,1],[-a,a,1]])
    Mdet = np.linalg.det(M)
    Mt = np.transpose(M)
    if Mdet != 0:
        Minv = np.linalg.inv(M)
    else:
        Minv = np.NaN
    return Minv, Mt, Mdet

def custom_matrix(m:int, n:int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    if m > 0 and n > 0 and isinstance(m, int) and isinstance(n, int):
        M = np.zeros((m, n))
        for i in range (0, m):
            for j in range (0, n):
                if i > j:
                    M[i][j] = i
                else:
                    M[i][j] = j
    else:
        return None
    return M