import numpy as np
from scipy import sparse

MAX_ITERATIONS = 1000
BETA = 0.85
STOP_THRESHOLD = 0.01


def input_m():
    n = int(input('Enter Matrix Size: '))

    l_matrix_rows = []
    l_matrix_columns = []

    degree_list = [0] * n

    print('!!! enter -1 to end')

    i = 0
    while True:
        s = input('edge {0}: '.format(i))

        if s == 'e':
            break

        node1, node2 = s.split(',')
        node1 = int(node1)
        node2 = int(node2)

        if node1 >= n or node1 < 0 or node1 == node2:
            print('!!! Wrong Node Number')
            continue

        l_matrix_rows.append(node1)
        l_matrix_columns.append(node2)

        degree_list[node1] += 1

        i += 1

    matrix_data = [1.0] * len(l_matrix_rows)

    for i, node in enumerate(l_matrix_rows):
        matrix_data[i] = 1 / degree_list[node]

    m_matrix = sparse.csr_matrix((matrix_data, (l_matrix_rows, l_matrix_columns)), shape=(n, n))

    return m_matrix.todense().T


def pagerank(m_matrix):
    """
    computes pagerank algorithm on input transition matrix
    :param m_matrix: transition matrix
    :return:
    """
    n = m_matrix.shape[0]

    ranks = np.ones((n, 1)) / n  # initial ranks is same as e/n
    last_ranks = ranks.copy()

    bm_matrix = BETA * m_matrix  # b*M
    bteleport_matrix = (1 - BETA) * (np.ones((n, 1)) / n)  # (1-b) * (e/n)
    for i in range(MAX_ITERATIONS):
        ranks = (bm_matrix.dot(ranks)) + bteleport_matrix  # v' = bMv + (1-b)e/n
        ranks = ranks / ranks.sum()

        err = np.abs(last_ranks - ranks)
        if err.all() <= STOP_THRESHOLD:
            break

        last_ranks = ranks.copy()

    return ranks  # a n * 1 vector showing pagerank of each node (ith element shows ith node pagerank)


def main():
    m_matrix = input_m()

    prank = pagerank(m_matrix)
    print(prank)


if __name__ == '__main__':
    main()
