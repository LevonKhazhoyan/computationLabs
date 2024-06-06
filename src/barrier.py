from scipy.optimize import minimize

def barrier_to_inf(value):
    if value == 0.0:
        return float('inf')
    return -1 / value


def barrier_b(x, phis):
    values = [phi(x) for phi in phis]
    return sum([barrier_to_inf(value) for value in values])


def barrier_theta(x, f, mu, phis):
    return f(x) + mu * barrier_b(x=x, phis=phis)


def barrier_check_range(x, phis):
    for phi in phis:
        if phi(x) > 0.0:
            return False
    return True


def barrier_solver(f, x_0, phis, mu, eps, coef):
    theta = lambda x: barrier_theta(x=x, f=f, mu=mu, phis=phis)
    b = lambda x: barrier_b(x=x, phis=phis)
    iteration = 0
    while iteration < 10_000 and mu * b(x_0) >= eps:
        new_x = minimize(theta, x_0).x
        if barrier_check_range(x=new_x, phis=phis):
            x_0 = new_x
        else:
            return x_0, iteration
        iteration += 1
        mu *= coef

    return x_0, iteration


def barrier_print(f, x_0, phis, mu, eps, coef):
    x, iteration = barrier_solver(f=f, x_0=x_0, phis=phis, mu=mu, eps=eps, coef=coef)
    print('Метод барьерных функций')
    print(f'Количество итераций: {iteration}')
    print(f'Ответ: {x}\n')