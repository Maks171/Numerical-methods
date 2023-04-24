import numpy as np
import scipy
import pickle
import typing
import math
import types
import pickle 
from inspect import isfunction


from typing import Union, List, Tuple

def fun(x):
    return np.exp(-2*x)+x**2-1

def dfun(x):
    return -2*np.exp(-2*x) + 2*x

def ddfun(x):
    return 4*np.exp(-2*x) + 2


def bisection(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą bisekcji.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and isinstance(epsilon, float) and isinstance(iteration, int) and callable(f):
        if f(a) * f(b) < 0:
            for i in range(iteration):
                c = a + (b - a) / 2
                if f(a) * f(c) < 0:
                    b = c
                    a = a
                elif f(a) * f(c) > 0:
                    a = c
                    b = b
                if np.abs(f(c)) < epsilon:
                    return c, i
            return a, iteration
    return None


def secant(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą siecznych.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and isinstance(epsilon, float) and isinstance(iteration, int):
        if epsilon > 0 and iteration > 0:
            if f(a) * f(b) < 0:
                for i in range(iteration):
                    x = (f(b) * a - f(a) * b) / (f(b) - f(a))
                    if f(a) * f(x) <= 0:
                        b = x
                    elif f(a) * f(x) > 0:
                        a = x
                    if abs(b - a) < epsilon or abs(f(x)) < epsilon:
                        return x, i
                return (f(b) * a - f(a) * b) / (f(b) - f(a)), iteration
    return None

def newton(f: typing.Callable[[float], float], df: typing.Callable[[float], float], ddf: typing.Callable[[float], float], a: Union[int,float], b: Union[int,float], epsilon: float, iteration: int) -> Tuple[float, int]:
    ''' Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.
    Parametry: 
    f - funkcja dla której jest poszukiwane rozwiązanie
    df - pochodna funkcji dla której jest poszukiwane rozwiązanie
    ddf - druga pochodna funkcji dla której jest poszukiwane rozwiązanie
    a - początek przedziału
    b - koniec przedziału
    epsilon - tolerancja zera maszynowego (warunek stopu)
    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and isinstance(epsilon, float) and isinstance(iteration, int) and(callable(i) for i in [f, df, ddf]):
        a_b = np.linspace(a, b, 1000)
        df_value = df(a_b)
        ddf_value = ddf(a_b)
        if not ((np.all(np.sign(df_value) < 0) or np.all(np.sign(df_value) > 0)) and
                (np.all(np.sign(ddf_value) < 0) or np.all(np.sign(ddf_value) > 0))):
            return None
        if f(a) * ddf(a) > 0:
            x = a
        else:
            x = b
        if f(a) * f(b) < 0:
            for i in range(iteration):
                xi = x - f(x) / df(x)
                if np.abs(xi - x) < epsilon:
                    return xi, i

                if i == iteration - 1:
                    return xi, iteration
                x = xi
    return None

