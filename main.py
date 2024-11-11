import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
from scipy.optimize import minimize
from mamdani import mamnadi_check

def least_squares_loss(params, X, y):
    predictions = np.dot(X, params)
    return np.sum((y - predictions) ** 2)


def gradient_descent_interpolation(X, y, learning_rate=0.01, num_iterations=1000):
    num_features = X.shape[1]
    params = np.random.rand(num_features)
    for i in range(num_iterations):
        gradients = -2 * np.dot(X.T, (y - np.dot(X, params))) / len(y)
        params -= learning_rate * gradients
    return params


def interpolate_with_least_squares(X, y):
    initial_params = np.random.rand(X.shape[1])
    result = minimize(least_squares_loss, initial_params, args=(X, y), method='BFGS')
    return result.x


def interpolate_data(data, method='least_squares'):
    X = data[['battery_power', 'ram', 'px']].values
    y = data['price_range'].values

    if method == 'least_squares':
        params = interpolate_with_least_squares(X, y)
    elif method == 'gradient_descent':
        params = gradient_descent_interpolation(X, y)
    else:
        raise ValueError("Unknown interpolation method")

    predictions = np.dot(X, params)
    return predictions


def plot_fuzzy_sets(antecedents):
    for var in antecedents:
        plt.figure(figsize=(10, 5))
        for term in var.terms:
            plt.plot(var.universe, var[term].mf, label=term)
        plt.title(f'Fuzzy Sets for {var.label}')
        plt.xlabel(var.label)
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid()
        plt.show()


def plot_results(results, predicted_price_range):
    plt.figure(figsize=(10, 5))

    plt.plot(results, marker='o', linestyle='-', color='b', label='Price Range Result')

    plt.plot(predicted_price_range, marker='x', linestyle='--', color='r', label='Predicted Price Range')

    plt.title('Comparison of Price Range Computation and Predicted Price Range')
    plt.xlabel('Sample Index')
    plt.ylabel('Price Range')
    plt.xticks(ticks=np.arange(len(results)), labels=np.arange(len(results)))
    plt.grid()
    plt.legend()
    plt.show()


def mamnadi_start(selected_func, num_intervals, data_from_csv, intervals_data, method='least_squares', test_data=None):
    array_battery_power = data_from_csv["battery_power"].to_numpy()
    array_ram = data_from_csv["ram"].to_numpy()
    array_px = data_from_csv["px"].to_numpy()
    array_price_range = data_from_csv["price_range"].to_numpy()

    if np.any(np.isnan(array_battery_power)) or np.any(np.isnan(array_ram)) or np.any(np.isnan(array_px)) or np.any(
            np.isnan(array_price_range)):
        print("NaN detected in input data.")
        return None

    battery_power = ctrl.Antecedent(np.linspace(array_battery_power.min(), array_battery_power.max(), 667),
                                    'battery_power')
    ram = ctrl.Antecedent(np.linspace(array_ram.min(), array_ram.max(), 667), 'ram')
    px = ctrl.Antecedent(np.linspace(array_px.min(), array_px.max(), 667), 'px')

    price_range = ctrl.Consequent(np.linspace(array_price_range.min(), array_price_range.max(), 100), 'price_range')

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
            elif selected_func == 'quadratic':
                var[term] = fuzz.pimf(var.universe, start, (start + end) / 2, end, end)

            print(f"{var_name} - {term}: {var[term]}")


    plot_fuzzy_sets(variables[:-1])


    predicted_price_range = interpolate_data(data_from_csv, method=method)

    data_from_csv['predicted_price_range'] = predicted_price_range

    data_from_csv.to_csv('predicted_price_range_output.csv', index=False)

    rules = []
    for bp_term in intervals_data['battery_power']:
        for ram_term in intervals_data['ram']:
            for px_term in intervals_data['px']:
                for pr_term in intervals_data['price_range']:
                    rule = ctrl.Rule(
                        battery_power[bp_term] & ram[ram_term] & px[px_term],
                        price_range[pr_term]
                    )
                    rules.append(rule)

    with open("fuzzy_rules.txt", "w") as file:
        for rule in rules:
            file.write(str(rule) + "\n")

    price_control_system = ctrl.ControlSystem(rules)

    X_train = data_from_csv[['battery_power', 'ram', 'px']].values
    y_train = data_from_csv['price_range'].values

    if method == 'least_squares':
        params = interpolate_with_least_squares(X_train, y_train)
    elif method == 'gradient_descent':
        params = gradient_descent_interpolation(X_train, y_train)

    if test_data is not None:
        results = []

        predicted_price_range_for_test = []

        for index, row in test_data.iterrows():
            price_control_simulation = ctrl.ControlSystemSimulation(price_control_system)

            price_control_simulation.input['battery_power'] = row["battery_power"]
            price_control_simulation.input['ram'] = row["ram"]
            price_control_simulation.input['px'] = row["px"]

            try:
                price_control_simulation.compute()
                result = price_control_simulation.output['price_range']
                print(f"Результат для строки {index}: {result}")
                results.append(result)
                predicted_price_range_for_test.append(result)

            except Exception as e:
                print(f"Ошибка при вычислении для строки {index}: {e}")
                results.append(None)
                predicted_price_range_for_test.append(None)


        X_test = test_data[['battery_power', 'ram', 'px']].values
        predicted_price_range_for_test = np.dot(X_test, params)

        test_data['predicted_price_range'] = predicted_price_range_for_test

        test_data.to_csv('test_with_predictions.csv', index=False)

        plot_results(results, predicted_price_range_for_test)

    return results


if __name__ == '__main__':

    selected_func = 'gaussmf'

    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    predic_test = pd.read_csv('test_with_predictions.csv')

    num_intervals = 3
    intervals_data = {
        'battery_power': ['Very Low', 'Low', 'Medium'],
        'ram': ['Very Low', 'Low', 'Medium'],
        'px': ['Very Low', 'Low', 'Medium'],
        'price_range': ['Very Cheap', 'Cheap', 'Optimal']
    }

    mamnadi_start(selected_func, num_intervals, train_data, intervals_data, method='least_squares', test_data=test_data)

    # intervals_data = {
    #     'battery_power': ['Very Low', 'Low', 'Medium'],
    #     'ram': ['Very Low', 'Low', 'Medium'],
    #     'px': ['Very Low', 'Low', 'Medium'],
    #     'predicted_price_range': ['Very Cheap', 'Cheap', 'Optimal']
    # }
    # mamnadi_check(selected_func, num_intervals, predic_test, intervals_data)



    # num_intervals = 5
    # intervals_data = {
    #     'battery_power': ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
    #     'ram': ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
    #     'px': ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
    #     'price_range': ['Very Cheap', 'Cheap', 'Optimal', 'Expensive', 'Very Expensive']
    # }
    # mamnadi_start(selected_func, num_intervals, train_data, intervals_data)
    #
    # num_intervals = 7
    # intervals_data = {
    #     'battery_power': ['Extremely Low', 'Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extremely High'],
    #     'ram': ['Extremely Low', 'Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extremely High'],
    #     'px': ['Extremely Low', 'Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extremely High'],
    #     'price_range': ['Very Cheap', 'Cheap', 'Optimal', 'Expensive', 'Very Expensive', 'Luxury', 'Ultra Luxury']
    # }
    # mamnadi_start(selected_func, num_intervals, train_data, intervals_data)
    #
    # num_intervals = 9
    # intervals_data = {
    #     'battery_power': ['Minimal', 'Very Low', 'Low', 'Below Average', 'Average', 'Above Average', 'High',
    #                       'Very High', 'Maximum'],
    #     'ram': ['Minimal', 'Very Low', 'Low', 'Below Average', 'Average', 'Above Average', 'High', 'Very High',
    #             'Maximum'],
    #     'px': ['Minimal', 'Very Low', 'Low', 'Below Average', 'Average', 'Above Average', 'High', 'Very High',
    #            'Maximum'],
    #     'price_range': ['Bargain', 'Very Cheap', 'Cheap', 'Optimal', 'Expensive', 'Very Expensive', 'Luxury',
    #                     'Premium', 'Ultra Premium']
    # }
    # mamnadi_start(selected_func, num_intervals, train_data, intervals_data)
