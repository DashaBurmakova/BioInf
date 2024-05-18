import numpy as np


def integrate_ode(start, end, step_size, max_function_calls, tolerance, differential_eq, initial_state):
    """
    Функция для численного решения ОДУ методом Рунге-Кутты с адаптивным шагом.

    Args:
        start (float): Начальное время.
        end (float): Конечное время.
        step_size (float): Начальный шаг интегрирования.
        max_function_calls (int): Максимальное количество вызовов функции.
        tolerance (float): Желаемая точность.
        differential_eq (callable): Функция, описывающая систему ОДУ.
        initial_state (list): Начальное состояние системы.

    """
    current_time = start
    state_vector = np.array(initial_state)
    function_calls = [0]

    print(f"{current_time:13.6f}{step_size:13.6f}{0:13d}{0:13d}", *[f"{x:12.6f}" for x in state_vector])

    def rk4_step(time, state, h):
        k1 = differential_eq(time, state, function_calls)
        k2 = differential_eq(time + h / 2, state + h * k1 / 2, function_calls)
        k3 = differential_eq(time + h / 2, state + h * k2 / 2, function_calls)
        k4 = differential_eq(time + h, state + h * k3, function_calls)
        return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    while current_time < end and function_calls[0] < max_function_calls:
        single_step = rk4_step(current_time, state_vector, step_size)
        half_step1 = rk4_step(current_time, state_vector, step_size / 2)
        half_step2 = rk4_step(current_time + step_size / 2, half_step1, step_size / 2)

        error_estimate = np.linalg.norm(half_step2 - single_step) / 15
        while error_estimate > tolerance:
            step_size /= 2
            single_step = rk4_step(current_time, state_vector, step_size)
            half_step1 = rk4_step(current_time, state_vector, step_size / 2)
            half_step2 = rk4_step(current_time + step_size / 2, half_step1, step_size / 2)
            error_estimate = np.linalg.norm(half_step2 - single_step) / 15

        if error_estimate < tolerance / 64:
            step_size *= 2

        current_time += step_size
        state_vector = single_step

        print(f"{current_time:13.6f}{step_size:13.6f}{error_estimate:13.5e}{function_calls[0]:13d}",
              *[f"{x:12.6f}" for x in state_vector])


# Получаем вводные данные
start_time = float(input())
end_time = float(input())
initial_step = float(input())
max_calls = int(input())
desired_accuracy = float(input())
num_equations = int(input())

# Считываем функции, описывающие систему ОДУ
func_code = []
for _ in range(num_equations + 3):
    line = input()
    func_code.append(line)

# Объединяем строки в один блок кода
code_block = '\n'.join(func_code)

# Начальное состояние системы
initial_conditions = list(map(float, input("Введите начальные условия: ").split()))

# Выполнение кода для определения функции fs
exec(code_block)

# Запуск интегратора ОДУ
integrate_ode(start_time, end_time, initial_step, max_calls, desired_accuracy, fs, initial_conditions)
