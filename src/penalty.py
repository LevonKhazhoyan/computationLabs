import numpy as np
from scipy.optimize import minimize


def penalty_phi(x, phis, p):
    return sum(max(0, phi(x)) ** p for phi in phis)


def penalty_psi(x, psis):
    return sum(np.abs(psi(x)) for psi in psis)


def penalty_h_x(x, phis, psis, p):
    return penalty_phi(x=x, phis=phis, p=p) + penalty_psi(x=x, psis=psis)


def penalty_theta(f, x, phis, psis, p, alpha):
    h_x = penalty_phi(x=x, phis=phis, p=p) + penalty_psi(x=x, psis=psis)
    return f(x) + alpha * penalty_h_x(x=x, phis=phis, psis=psis, p=p)


def penalty_check_range(x, phis, psis):
    for phi in phis:
        if phi(x) <= 0:
            return False
    for psi in psis:
        if psi(x) == 0:
            return False
    return True


def penalty_solver(f, phis, psis, x_0, p, eps, coef, alpha):
    theta = lambda x: penalty_theta(f=f, x=x, phis=phis, psis=psis, p=p, alpha=alpha)
    h_x = lambda x: penalty_h_x(x=x, phis=phis, psis=psis, p=p)
    iteration = 0
    while alpha * h_x(x_0) >= eps and iteration < 10_000:
        x_0 = minimize(theta, x_0).x
        alpha *= coef
        iteration += 1
    return x_0, iteration


def penalty_print(f, phis, psis, x_0, p, eps, coef, alpha):
    x, iteration = penalty_solver(f=f, psis=psis, phis=phis, alpha=alpha, x_0=x_0, eps=eps, p=p, coef=coef)
    print('Метод штрафных функций')
    print(f'Количество итераций: {iteration}')
    print(f'Ответ: {x}\n')