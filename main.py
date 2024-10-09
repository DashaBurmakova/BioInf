import numpy as np

def integrate_ode(start, end, step_size, max_function_calls, tolerance, differential_eqs, initial_state):
    """
    Функция для численного решения ОДУ методом Рунге-Кутты с адаптивным шагом.
    """
    current_time = start
    state_vector = np.array(initial_state)
    function_calls = [0]

    print(f"{current_time:13.6f}{step_size:13.6f}{0:13d}{0:13d}", *[f"{x:12.6f}" for x in state_vector])

    def rk4_step(time, state, h):
        k1 = np.array([eq(time, state, function_calls) for eq in differential_eqs])
        k2 = np.array([eq(time + h / 2, state + h * k1 / 2, function_calls) for eq in differential_eqs])
        k3 = np.array([eq(time + h / 2, state + h * k2 / 2, function_calls) for eq in differential_eqs])
        k4 = np.array([eq(time + h, state + h * k3, function_calls) for eq in differential_eqs])
        return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    while current_time < end and function_calls[0] < max_function_calls:
        # Вычисляем k1 и single_step
        k1 = np.array([eq(current_time, state_vector, function_calls) for eq in differential_eqs])
        single_step = state_vector + (step_size / 6) * k1

        # Вычисляем half_step1 и k2
        half_step1 = rk4_step(current_time, state_vector, step_size / 2)
        k2 = np.array([eq(current_time + step_size / 2, half_step1, function_calls) for eq in differential_eqs])

        # Вычисляем half_step2 и k3
        half_step2 = rk4_step(current_time + step_size / 2, half_step1, step_size / 2)
        k3 = np.array([eq(current_time + step_size / 2, half_step2, function_calls) for eq in differential_eqs])

        # Вычисляем k4
        k4 = np.array([eq(current_time + step_size, half_step2, function_calls) for eq in differential_eqs])

        # Оценка ошибки
        error_estimate = np.linalg.norm(half_step2 - (state_vector + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4))) / 15

        while error_estimate > tolerance and function_calls[0] < max_function_calls:
            step_size /= 2
            k1 = np.array([eq(current_time, state_vector, function_calls) for eq in differential_eqs])
            single_step = state_vector + (step_size / 6) * k1

            half_step1 = rk4_step(current_time, state_vector, step_size / 2)
            k2 = np.array([eq(current_time + step_size / 2, half_step1, function_calls) for eq in differential_eqs])
            half_step2 = rk4_step(current_time + step_size / 2, half_step1, step_size / 2)
            k3 = np.array([eq(current_time + step_size / 2, half_step2, function_calls) for eq in differential_eqs])
            k4 = np.array([eq(current_time + step_size, half_step2, function_calls) for eq in differential_eqs])

            error_estimate = np.linalg.norm(half_step2 - (state_vector + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4))) / 15

        if error_estimate < tolerance / 64:
            step_size *= 2

        current_time += step_size
        state_vector = half_step2  # Используем half_step2 для следующей итерации

        print(f"{current_time:13.6f}{step_size:13.6f}{error_estimate:13.5e}{function_calls[0]:13d}",
              *[f"{x:12.6f}" for x in state_vector])

# Получаем вводные данные
start_time = float(input("Введите начальное время: "))
end_time = float(input("Введите конечное время: "))
initial_step = float(input("Введите начальный шаг интегрирования: "))
max_calls = int(input("Введите максимальное количество вызовов функции: "))
desired_accuracy = float(input("Введите желаемую точность: "))
num_equations = int(input("Введите количество уравнений: "))

# Чтение уравнений
equations = []
for i in range(num_equations):
    equation = input(f"Введите выражение для уравнения {i + 1} (например, '-0.5 * y[0]'): ")
    # Создание замыкания для каждого уравнения
    equations.append(lambda t, y, calls, eq=equation: eval(eq))

# Начальное состояние системы
initial_conditions = list(map(float, input("Введите начальные условия: ").split()))

# Запуск интегратора ОДУ
integrate_ode(start_time, end_time, initial_step, max_calls, desired_accuracy, equations, initial_conditions)
