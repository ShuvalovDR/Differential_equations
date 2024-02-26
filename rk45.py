import numpy as np


def rk45(x_0: float, x_1: float, y_0: np.array, n: int, a: np.array, w=np.array([0, 0, 0, 0])):
    """
    :param x_0: Начало отрезка интегрирования.
    :param x_1: Конец отрезка интегрирования
    :param y_0: Начальные условия
    :param n: Количество точек разбиения отрезка
    :param a: Матрица системы
    :param w: Вектор возмущений
    :return: Сетка и решение на сетке
    """
    grid = np.linspace(x_0, x_1, n)
    h = (x_1 - x_0) / (n - 1)
    solution = np.zeros((4, n))
    solution[:, 0] = y_0
    for i in range(1, n):
        k1 = a @ solution[:, i - 1] + w
        k2 = a @ (solution[:, i - 1] + h / 2 * k1) + w
        k3 = a @ (solution[:, i - 1] + h / 2 * k2) + w
        k4 = a @ (solution[:, i - 1] + h * k3) + w
        solution[:, i] = solution[:, i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return grid, solution
