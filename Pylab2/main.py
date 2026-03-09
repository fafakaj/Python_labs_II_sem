import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_vector():
    """
    Создать массив от 0 до 9.

    Returns:
        numpy.ndarray: Массив чисел от 0 до 9 включительно
    """
    return  np.arange(10)


def create_matrix():
    """
    Создать матрицу 5x5 со случайными числами [0,1].

    Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями от 0 до 1
    """
    return np.random.rand(5,5)


def reshape_vector(vec):
    """
    Преобразовать (10,) -> (2,5)

    Args:
        vec (numpy.ndarray): Входной массив формы (10,)

    Returns:
        numpy.ndarray: Преобразованный массив формы (2, 5)
    """
    return vec.reshape(2,5)


def transpose_matrix(mat):
    """
    Транспонирование матрицы.

    Args:
        mat (numpy.ndarray): Входная матрица

    Returns:
        numpy.ndarray: Транспонированная матрица
    """
    return np.transpose(mat)


def vector_add(a, b):
    """
    Сложение векторов одинаковой длины.
    (Векторизация без циклов)

    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор

    Returns:
        numpy.ndarray: Результат поэлементного сложения
    """
    if np.shape(a) != np.shape(b):
        raise ValueError (f'Векторы разной длины.')

    return a + b


def scalar_multiply(vec, scalar):
    """
    Умножение вектора на число.

    Args:
        vec (numpy.ndarray): Входной вектор
        scalar (float/int): Число для умножения

    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр
    """
    if not isinstance(scalar, (float, int)):
        raise TypeError( f"scalar должен быть числом, а не {type(scalar)}.")

    return vec * scalar


def elementwise_multiply(a, b):
    """
    Поэлементное умножение.

    Args:
        a (numpy.ndarray): Первый вектор/матрица
        b (numpy.ndarray): Второй вектор/матрица

    Returns:
        numpy.ndarray: Результат поэлементного умножения
    """
    if a.shape != b.shape:
        raise ValueError(f"Матрицы/вектора должны иметь одинаковые размеры.")

    return a * b

def dot_product(a, b):
    """
    Скалярное произведение.

    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор

    Returns:
        float: Скалярное произведение векторов
    """
    return np.dot(a,b)


def matrix_multiply(a, b):
    """
    Умножение матриц.

    Args:
        a (numpy.ndarray): Первая матрица
        b (numpy.ndarray): Вторая матрица

    Returns:
        numpy.ndarray: Результат умножения матриц
    """
    shape_a = a.shape
    shape_b = b.shape

    if shape_a[-1] != shape_b[-2]:
        raise ValueError(f"Неправильный размер матриц для умножения.")

    return a @ b


def matrix_determinant(a):
    """
    Определитель матрицы.

    Args:
        a (numpy.ndarray): Квадратная матрица

    Returns:
        float: Определитель матрицы
    """
    a_shape = a.shape
    if a_shape[-1] != a_shape[-2]:
        raise ValueError(f"Детерминант определен только для квадратной матрицы.")
    return np.linalg.det(a)


def matrix_inverse(a):
    """
    Обратная матрица.

    Args:
        a (numpy.ndarray): Квадратная матрица

    Returns:
        numpy.ndarray: Обратная матрица
    """
    a_shape = a.shape
    if a_shape[-1] != a_shape[-2]:
        raise ValueError(f"Детерминант определен только для квадратной матрицы.")

    if np.isclose(np.linalg.det(a), 0):
        raise ValueError(f"Детерминант должен быть отличен от нуля.")

    return np.linalg.inv(a)


def solve_linear_system(a, b):
    """
    Решить систему Ax = b

    Args:
        a (numpy.ndarray): Матрица коэффициентов A
        b (numpy.ndarray): Вектор свободных членов b

    Returns:
        numpy.ndarray: Решение системы x
    """
    a_shape = a.shape
    if a_shape[-1] != a_shape[-2]:
        raise ValueError(f"Детерминант определен только для квадратной матрицы.")

    if np.isclose(np.linalg.det(a), 0):
        raise ValueError(f"Детерминант должен быть отличен от нуля.")

    return np.linalg.solve(a,b)


def load_dataset(path="data/students_scores.csv"):
    """
    Загрузить CSV и вернуть NumPy массив.

    Args:
        path (str): Путь к CSV файлу

    Returns:
        numpy.ndarray: Загруженные данные в виде массива
    """
    return pd.read_csv(path).to_numpy()

def statistical_analysis(data):
    """
    Представьте, что данные — это результаты экзамена по математике.
    Нужно оценить:
    - средний балл
    - медиану
    - стандартное отклонение
    - минимум
    - максимум
    - 25 и 75 перцентили

    Args:
        data (numpy.ndarray): Одномерный массив данных

    Returns:
        dict: Словарь со статистическими показателями
    """
    return {"mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "percentile_25": np.percentile(data,25),
            "percentile_75": np.percentile(data,75)
            }


def normalize_data(data):
    """
    Min-Max нормализация.

    Формула: (x - min) / (max - min)

    Args:
        data (numpy.ndarray): Входной массив данных

    Returns:
        numpy.ndarray: Нормализованный массив данных в диапазоне [0, 1]
    """
    if np.max(data) == np.min(data):
        raise ValueError(f"Нельзя делить на ноль.")

    return (data - np.min(data))/(np.max(data) - np.min(data))


def plot_histogram(data):
    """
    Построить гистограмму распределения оценок по математике.

    Args:
        data (numpy.ndarray): Данные для гистограммы
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data)
    plt.title("Распределение оценок по математике")
    plt.xlabel("Оценка")
    plt.ylabel("Количество студентов")

    mean_val = np.mean(data)
    plt.axvline(mean_val, color = 'red', label = f'Среднее: {mean_val}')

    plt.legend()
    plt.savefig('plots/math_scores_hist.png')
    plt.close()


def plot_heatmap(matrix):
    """
    Построить тепловую карту корреляции предметов.

    Args:
        matrix (numpy.ndarray): Матрица корреляции
    """
    plt.figure(figsize=(10,8))
    sns.heatmap(matrix, annot=True, fmt='.2f', linewidths=1, linecolor='gray', cbar_kws={'label': 'Коэффициент корреляции'})
    plt.title('Тепловая карта корреляции')

    plt.savefig('plots/correlation_heatmap.png')
    plt.close()


def plot_line(x, y):
    """
    Построить график зависимости: студент -> оценка по математике.

    Args:
        x (numpy.ndarray): Номера студентов
        y (numpy.ndarray): Оценки студентов
    """
    plt.plot(x, y, color='darkblue', linewidth=2)
    plt.title('График зависимости: студент -> оценка по математике')
    plt.xlabel('Номера студентов')
    plt.ylabel('Оценки студентов')

    plt.savefig('plots/student-marks.png')
    plt.close()
