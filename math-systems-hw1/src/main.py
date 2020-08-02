import numpy as np

CONVERGENCE_THRESHOLD = 1  # in percent
SOR_LANDA = 1.2
EPSILON = 0.00001


def get_equations():
    """
    gets equations from input matrix
    :return: [[a11, a12, a13, ...], [a21, a22, ...]], [y1, y2, y3, ...]
    """
    n = int(input('Enter Variables Count: '))

    all_coefficients = []
    all_y = []

    print('Example Equation: "a1,a2,...,an,b')
    for i in range(n):
        values = input('Input {0}th Equation Values: '.format(i))

        values = values.split(',')
        values = [float(x) for x in values]

        coefficients = values[0:n]
        y = values[n]

        all_coefficients.append(coefficients)
        all_y.append([y])

    a = np.array(all_coefficients)
    b = np.array(all_y)

    return a, b


def rearrange_ab(a, b):
    """
    rearranges a and b to become diagonally dominent
    :param a:
    :param b:
    :return:
    """
    n = a.shape[0]
    permutation = [-1] * n
    for i in range(n):
        row = a[i, :]
        for j in range(n):
            summation = row.sum() - row[j]
            if row[j] >= summation:
                if j not in permutation:
                    permutation[i] = j

    if -1 in permutation:
        return None, None

    rearranged_a = np.zeros(a.shape)
    rearranged_b = np.zeros(b.shape)

    for i, p in enumerate(permutation):
        rearranged_a[p, :] = a[i, :]
        rearranged_b[p, :] = b[i, :]

    return rearranged_a, rearranged_b


def generate_cd_matrices(a, b):
    """
    generates c and d matrices from a and b
    :param a:
    :param b:
    :return: c, d
    """
    n = a.shape[0]
    c = a.copy()
    d = b.copy()
    for i in range(n):
        denom = a[i, i] if a[i, i] > 0 else EPSILON
        c[i, i] = 0
        c[i, :] *= 1 / denom
        d[i, :] *= 1 / denom

    return c, d


def check_convergence(x_old, x_new):
    """
    checks if two vectors have converged or not
    :param x_old:
    :param x_new:
    :return:
    """
    n = x_old.shape[0]

    for i in range(n):
        old_val = x_old[i, 0]
        new_val = x_new[i, 0]

        if old_val != 0:
            diff = abs((old_val - new_val) / old_val) * 100
        else:
            diff = abs((old_val - new_val) / EPSILON) * 100

        if diff > CONVERGENCE_THRESHOLD:
            return False

    return True


def jacobi(c, d):
    """
    solve equation with jacobi method
    :param n:
    :param c:
    :param d:
    :return:
    """
    n = c.shape[0]
    x_old = np.zeros(shape=(n, 1))
    x_new = np.zeros(shape=(n, 1))

    while True:
        for i in range(n):
            current_x = d[i, 0]

            for j in range(n):
                current_x = current_x - (c[i, j] * x_old[j, 0])

            x_new[i, 0] = current_x

        if check_convergence(x_old, x_new):
            break

        x_old = x_new.copy()

    return x_new


def gauss_seidel(c, d):
    """
    solve equation with gauss-seidel method
    :param c:
    :param d:
    :return:
    """
    n = c.shape[0]
    x_old = np.zeros(shape=(n, 1))
    x_new = np.zeros(shape=(n, 1))

    while True:
        for i in range(n):
            current_x = d[i, 0]

            for j in range(n):
                if j <= i:
                    current_x = current_x - (c[i, j] * x_new[j, 0])
                else:
                    current_x = current_x - (c[i, j]) * x_old[j, 0]

            x_new[i, 0] = current_x

        if check_convergence(x_old, x_new):
            break

        x_old = x_new.copy()

    return x_new


def sor(c, d, landa=SOR_LANDA):
    """
    solve equation with sor method
    :param landa:
    :param c:
    :param d:
    :return:
    """
    n = c.shape[0]
    x_old = np.zeros(shape=(n, 1))
    x_new = np.zeros(shape=(n, 1))

    while True:
        for i in range(n):
            current_x = ((1 - landa) * x_old[i, 0]) + (landa * d[i, 0])

            for j in range(n):
                if j <= i:
                    current_x = current_x - (landa * c[i, j] * x_new[j, 0])
                else:
                    current_x = current_x - (landa * c[i, j]) * x_old[j, 0]

            x_new[i, 0] = current_x

        if check_convergence(x_old, x_new):
            break

        x_old = x_new.copy()

    return x_new


def main():
    a, b = get_equations()

    a, b = rearrange_ab(a, b)

    if a is None or b is None:
        print('Cannot Make this Matrix Diagonally Dominant')
        return

    c, d = generate_cd_matrices(a, b)

    print('Method: ')
    print('1. Jacobi')
    print('2. Gauss-Seidel')
    print('3. Successive Over Relaxation (SOR)')
    algorithm = int(input('Enter Your Choice: '))

    if algorithm == 1:
        result = jacobi(c, d)
    elif algorithm == 2:
        result = gauss_seidel(c, d)
    elif algorithm == 3:
        landa = float(input('Enter Landa: '))
        result = sor(c, d, landa)
    else:
        result = '!!! Bad Choice'

    print(result)


if __name__ == '__main__':
    main()
