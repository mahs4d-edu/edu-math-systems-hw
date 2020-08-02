import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns

TOTAL_MOVES = 10000
T_PER_MOVE = 20

NETWORK_TYPE_ERDOS = 0
NETWORK_TYPE_WATTS = 1
NETWORK_ERDOS_P = 0.1
NETWORK_WATTS_K = 20
NETWORK_WATTS_P = 0.1
Q3_TOTAL_MOVES = 100
Q3_T_PER_MOVE = 200


def question1():
    all_transitions = []
    all_transitions_squares = []
    transition_square_means = []

    # do random walk TOTAL_MOVES times
    for i in range(TOTAL_MOVES):
        x = 0

        # the random walk algorithm
        for j in range(T_PER_MOVE):
            dx = random.choice([-1, 1])
            x += dx

        all_transitions.append(x)
        all_transitions_squares.append(x ** 2)

        # transition square means
        transition_square_means.append(np.mean(np.array(all_transitions_squares)))

    # diffusion coefficient is equal to the last transition square mean we computed
    diffusion_coefficient = transition_square_means[-1]
    print('Diffusion Coefficient: {0}'.format(diffusion_coefficient))

    # draw transition square means
    tsm_data = pd.DataFrame(data={'move': range(TOTAL_MOVES), 'tsm': transition_square_means,
                                  'diff_co': [diffusion_coefficient] * TOTAL_MOVES})
    sns.lineplot(x='move', y='tsm', data=tsm_data)
    sns.lineplot(x='move', y='diff_co', data=tsm_data)
    plt.show()

    # draw distribution
    sns.distplot(all_transitions, rug=True)
    plt.show()


def question2():
    all_transitions = []
    all_transitions_squares = []
    transition_square_means = []

    # do random walk TOTAL_MOVES times
    for i in range(TOTAL_MOVES):
        x = 0

        # the random walk algorithm
        for j in range(T_PER_MOVE):
            dx = random.choice([-1, 1])
            x += dx

            if x == -1:
                x = 0

        all_transitions.append(x)
        all_transitions_squares.append(x ** 2)

        # transition square means
        transition_square_means.append(np.mean(np.array(all_transitions_squares)))

    # diffusion coefficient is equal to the last transition square mean we computed
    diffusion_coefficient = transition_square_means[-1]
    print('Diffusion Coefficient: {0}'.format(diffusion_coefficient))

    # draw transition square means
    tsm_data = pd.DataFrame(data={'move': range(TOTAL_MOVES), 'tsm': transition_square_means,
                                  'diff_co': [diffusion_coefficient] * TOTAL_MOVES})
    sns.lineplot(x='move', y='tsm', data=tsm_data)
    sns.lineplot(x='move', y='diff_co', data=tsm_data)
    plt.show()

    # draw distribution
    sns.distplot(all_transitions, rug=True)
    plt.show()


def question3_sub(network_type):
    all_transitions = []
    all_transitions_squares = []
    transition_square_means = []

    network_type_str = 'Erdos-Renyi' if network_type == NETWORK_TYPE_ERDOS else 'Watts-Strogatz'

    if network_type == NETWORK_TYPE_ERDOS:
        graph = nx.erdos_renyi_graph(300, NETWORK_ERDOS_P)
    else:
        graph = nx.watts_strogatz_graph(300, NETWORK_WATTS_K, NETWORK_WATTS_P)

    # do random walk Q3_TOTAL_MOVES times
    for i in range(Q3_TOTAL_MOVES):
        x = 0

        # random walk algorithm
        for j in range(Q3_T_PER_MOVE):
            neighbors = list(graph.neighbors(x))
            x = random.choice(neighbors)

        # distance from origin
        x_distance = nx.shortest_path_length(graph, x, 0)

        all_transitions.append(x_distance)
        all_transitions_squares.append(x_distance ** 2)

        # transition square means
        transition_square_means.append(np.mean(np.array(all_transitions_squares)))

    # diffusion coefficient is equal to the last transition square mean we computed
    diffusion_coefficient = transition_square_means[-1]
    print('Diffusion Coefficient for {0} Network: {1}'.format(network_type_str, diffusion_coefficient))

    # draw transition square means
    tsm_data = pd.DataFrame(data={'move': range(Q3_TOTAL_MOVES), 'tsm': transition_square_means,
                                  'diff_co': [diffusion_coefficient] * Q3_TOTAL_MOVES})
    sns.lineplot(x='move', y='tsm', data=tsm_data)
    sns.lineplot(x='move', y='diff_co', data=tsm_data)
    plt.show()

    # draw distribution
    sns.distplot(all_transitions, rug=True)
    plt.show()


def question3():
    question3_sub(NETWORK_TYPE_ERDOS)
    question3_sub(NETWORK_TYPE_WATTS)


if __name__ == '__main__':
    q_funcs = [question1, question2, question3]
    q = 0

    if len(sys.argv) > 1:
        q = int(sys.argv[1]) - 1

    if not 0 <= q <= 2:
        print('Wrong Input')
        sys.exit(1)

    q_funcs[q]()
