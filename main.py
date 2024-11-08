import numpy as np
import skfuzzy as fuzz
import pandas as pd
from skfuzzy import control as ctrl
import json


def gradient_descent(X, y, learning_rate=0.001, iterations=1000):
    m = len(y)
    theta = np.zeros(X.shape[1])

    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (learning_rate / m) * (X.T.dot(errors))

        # Проверка на NaN
        if np.any(np.isnan(theta)):
            print("NaN detected in theta, stopping gradient descent.")
            break

    return theta


def least_squares(X, y):

    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best


def mamnadi_start(selected_func, num_intervals, data_from_csv, intervals_data):
    array_battery_power = data_from_csv["battery_power"].to_numpy()
    array_ram = data_from_csv["ram"].to_numpy()
    array_px = data_from_csv["px"].to_numpy()
    array_price_range = data_from_csv["price_range"].to_numpy()

    # Проверка на NaN в данных
    if np.any(np.isnan(array_battery_power)) or np.any(np.isnan(array_ram)) or np.any(np.isnan(array_px)) or np.any(np.isnan(array_price_range)):
        print("NaN detected in input data.")
        return None

    # Создаем входные переменные
    battery_power = ctrl.Antecedent(np.linspace(array_battery_power.min(), array_battery_power.max(), 667), 'battery_power')
    ram = ctrl.Antecedent(np.linspace(array_ram.min(), array_ram.max(), 667), 'ram')
    px = ctrl.Antecedent(np.linspace(array_px.min(), array_px.max(), 667), 'px')

    # Создаем выходную переменную с ТОЧНЫМ именем 'price_range'
    price_range = ctrl.Consequent(np.linspace(array_price_range.min(), array_price_range.max(), 100), 'price_range')

    # Определяем термы для входных и выходной переменной
    variables = [battery_power, ram, px, price_range]

    for var, (var_name, terms) in zip(variables, intervals_data.items()):
        intervals = np.linspace(var.universe.min(), var.universe.max(), num_intervals + 1)

        for term, (start, end) in zip(terms, zip(intervals, intervals[1:])):
            if selected_func == 'trimf':
                var[term] = fuzz.trimf(var.universe, [start, (start + end) / 2, end])
            elif selected_func == 'trapmf':
                var[term] = fuzz.trapmf(var.universe, [start, start + (end - start) / 4, end - (end - start) / 4, end])
            elif selected_func == 'gaussmf':
                var[term] = fuzz.gaussmf(var.universe, (start + end) / 2, (end - start) / 4)

            # Вывод значений функции принадлежности
            print(f"{var_name} - {term}: {var[term]}")

    # Создаем правила
    rules = []
    rules_json = []
    for bp_term in intervals_data['battery_power']:
        for ram_term in intervals_data['ram']:
            for px_term in intervals_data['px']:
                for pr_term in intervals_data['price_range']:
                    rule = ctrl.Rule(
                        battery_power[bp_term] & ram[ram_term] & px[px_term],
                        price_range[pr_term]
                    )
                    rules.append(rule)

                    rule_dict = {
                        'battery_power': bp_term,
                        'ram': ram_term,
                        'px': px_term,
                        'price_range': pr_term
                    }
                    rules_json.append(rule_dict)


    with open('fuzzy_rules.json', 'w') as f:
        json.dump(rules_json, f, indent=4)

    # Создаем систему управления
    price_control_system = ctrl.ControlSystem(rules)

    # Список для хранения результатов
    results = []

    for index, row in data_from_csv.iterrows():
        price_control_simulation = ctrl.ControlSystemSimulation(price_control_system)

        price_control_simulation.input['battery_power'] = row["battery_power"]
        price_control_simulation.input['ram'] = row["ram"]
        price_control_simulation.input['px'] = row["px"]

        try:
            price_control_simulation.compute()
            result = price_control_simulation.output['price_range']
            print(f"Результат для строки {index}: {result}")
            results.append(result)

        except Exception as e:
            print(f"Ошибка при вычислении для строки {index}: {e}")
            results.append(None)

    print("Итоговые результаты:", results)

    X = np.column_stack((array_battery_power, array_ram, array_px))
    theta_gd = gradient_descent(X, array_price_range)

    theta_ls = least_squares(X, array_price_range)

    print("Параметры градиентного спуска:", theta_gd)
    print("Параметры метода наименьших квадратов:", theta_ls)

    return results


if __name__ == '__main__':
    selected_func = 'trimf'
    num_intervals = 3

    train_data = pd.read_csv('train.csv')

    intervals_data = {
        'battery_power': ['Little', 'Medium', 'Much'],
        'ram': ['Little', 'Medium', 'Many'],
        'px': ['Little', 'Medium', 'Many'],
        'price_range': ['Cheap', 'Optimal', 'Expensive']
    }
    mamnadi_start(selected_func, num_intervals, train_data, intervals_data)