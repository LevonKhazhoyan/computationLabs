import numpy as np
from scipy.optimize import minimize

def lagrange_square(x):
    return (sum(np.power(x, 2)))

def lagrange_proection(values):
    return np.array([max(value, 0) for value in values])

def lagrange_m(f, x, a, our_lambda, phis):
    values = np.array([phi(x) for phi in phis])
    lambda_plus_values = lagrange_proection(our_lambda + a * values)
    return f(x) + 1 / (2 * a) * lagrange_square(lambda_plus_values) - 1 / (2 * a) * lagrange_square(our_lambda)


def lagrange_theta(f, x, x_k, alpha, a, our_lambda, phis):
    return 1 / 2 * lagrange_square(x - x_k) + alpha * lagrange_m(f=f, x=x, a=a, our_lambda=our_lambda, phis=phis)


def lagrange_check_range(x, phis):
    for phi in phis:
        if phi(x) > 0:
            return False
    return True


def lagrange_solver(f, phis, our_lambda, a, alpha, x_0, eps):
    theta = lambda x, x_k, lambda_k: lagrange_theta(f=f, x=x, x_k=x_k, alpha=alpha, a=a, our_lambda=lambda_k, phis=phis)

    x_k = x_0
    x_k_minus_1 = 0
    iteration = 0

    while (iteration == 0) or (iteration < 10_000 and np.sum(np.abs(x_k - x_k_minus_1)) >= eps):

        theta_k_minus_1 = lambda x: theta(x, x_k, our_lambda)
        x_k_minus_1 = x_k
        x_k = minimize(theta_k_minus_1, x_k).x

        values = np.array([phi(x_k) for phi in phis])
        our_lambda = lagrange_proection(our_lambda + a * values)
        iteration += 1
    return x_k, iteration

def lagrange_print(f, phis, our_lambda, a, alpha, x_0, eps):
    x, iteration = lagrange_solver(f, phis, our_lambda, a, alpha, x_0, eps)
    print('Метод модифицированных функций Лагранжа')
    print(f'Количество итераций: {iteration}')
    print(f'Ответ: {x}\n')