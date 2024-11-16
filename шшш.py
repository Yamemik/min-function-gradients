import numpy as np

# Определяем целевую функцию
def f(x, y):
    return x**2 + y**2 + 2*x*y  # Пример функции

# Определяем градиент функции
def gradient(x, y):
    df_dx = 2 * x + 2 * y
    df_dy = 2 * y + 2 * x
    return np.array([df_dx, df_dy])

# Реализация метода золотого сечения для нахождения шага
def golden_section_search(func, x0, d, a=-1, b=1, tol=1e-5):
    phi = (1 + np.sqrt(5)) / 2  # Золотое сечение
    resphi = 2 - phi

    # Начальные точки
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)

    while np.abs(b - a) > tol:
        if func(x0 + x1 * d[0], x0 + x1 * d[1]) < func(x0 + x2 * d[0], x0 + x2 * d[1]):
            b = x2
        else:
            a = x1
        
        # Обновление x1 и x2
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)

    return (a + b) / 2  # Оптимальный шаг

# Реализация метода наискорейшего спуска
def steepest_descent_with_golden_section(starting_point, tolerance, max_iterations):
    x_current = np.array(starting_point)
    
    for i in range(max_iterations):
        grad = gradient(x_current[0], x_current[1])  # Вычисляем градиент
        direction = -grad  # Направление спуска
        
        # Находим оптимальный шаг с помощью метода золотого сечения
        optimal_step = golden_section_search(lambda alpha: f(x_current[0] + alpha * direction[0],
                                                              x_current[1] + alpha * direction[1]),
                                              x_current,
                                              direction)
        
        # Обновляем текущее значение
        x_next = x_current + optimal_step * direction
        
        # Проверка условия сходимости
        if np.linalg.norm(grad) < tolerance:
            print(f"Сошлось за {i} итераций.")
            return x_next
        
        x_current = x_next  # Обновляем текущее значение
    
    print("Максимальное количество итераций превышено.")
    return x_current

# Начальная точка
starting_point = [3, 4]
# Порог сходимости
tolerance = 1e-6
# Максимальное количество итераций
max_iterations = 1000

# Запуск алгоритма
minimum = steepest_descent_with_golden_section(starting_point, tolerance, max_iterations)

print("Найденный минимум:", minimum)
