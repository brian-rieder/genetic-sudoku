#!/usr/bin/env python3.7

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from random import randint, sample, seed


############################################### GENETIC ALGORITHM ###############################################


def how_many_missing(num_array):
    flags = [1] * len(num_array)
    for num_to_check in range(1, len(num_array)+1):
        if num_to_check in num_array:
            flags[num_to_check-1] = 0
    return sum(flags)


def check_square(board, row, col):
    base = int(len(board) ** 0.5)
    # check errors in the row
    row_errors = how_many_missing(board[row])
    # check errors in the column
    xboard = np.array(board)
    column_errors = how_many_missing(xboard[:, col])
    # check errors in the box
    box_list = []
    mul_row = row // base
    mul_col = col // base
    for i in range(0, base):
        for j in range(0, base):
            row_idx = (col + j) % base + base * mul_col
            col_idx = (row + i) % base + base * mul_row
            box_list.append(board[row_idx][col_idx])
    box_errors = how_many_missing(box_list)
    return (row_errors + column_errors + box_errors) == 0


def fitness_function(board):
    """
    Determines the fitness value (i.e., how close it is to a final solution) of the given board. This is done via
    determination of the number of errors that exist in the current board.
    :param board: the board whose fitness is being evaluated
    :return: integer value representing the fitness of the given board
    """
    side_length = len(board)
    base = int(side_length ** 0.5)
    # check errors in each row
    row_sum = sum([how_many_missing(row) for row in board])
    # check errors in each column
    column_sum = sum([how_many_missing(column) for column in zip(*board)])
    # check errors in each interior box
    box_sum = 0
    for row_idx in range(0, base):
        for col_idx in range(0, base):
            box = [row[row_idx*base:(row_idx+1)*base] for row in board[col_idx*base:(col_idx+1)*base]]
            box_sum += how_many_missing(sum(box, []))
    # return the total number of errors
    return row_sum + column_sum + box_sum


def get_best_fitness(board_list):
    """
    Returns the best (lowest) fitness value of the boards being analyzed.
    :param board_list: the list of boards being analyzed
    :return: the float fitness value of the best board
    """
    return min([fitness_function(board) for board in board_list])


def get_best_board(board_list):
    """
    Returns the best board (i.e., the board that has the lowest fitness value)
    :param board_list: the list of boards being analyzed
    :return: the board the had the lowest fitness value
    """
    fitnesses = [fitness_function(board) for board in board_list]
    return board_list[fitnesses.index(min(fitnesses))]


def mutation_operator_old(board_list, removal_map):
    """
    Mutates the provided list of boards by incrementing the least mutated puzzle index in a random row.
    :param board_list: the list of puzzle boards
    :param removal_map: the mapping which indicates which indices need to be updated
    :return: updated list of mutated boards
    """
    board_size = len(board_list)
    for board in board_list:
        row = randint(0, board_size-1)
        col = removal_map[row].index(min(x for x in removal_map[row] if x > 0))
        board[row][col] = ((board[row][col] + 1) % board_size) + 1
        removal_map[row][col] += 1
    return board_list, removal_map


def mutation_operator(board_list, removal_map):
    """
    Mutates the provided list of boards by incrementing the least mutated puzzle index in a random row.
    :param board_list: the list of puzzle boards
    :param removal_map: the mapping which indicates which indices need to be updated
    :return: updated list of mutated boards
    """
    fitness_list = [fitness_function(board) for board in board_list]
    initial_fitness = max(fitness_list)
    worst_board = board_list[fitness_list.index(initial_fitness)]
    board_size = len(worst_board)
    row = randint(0, board_size-1)
    col = removal_map[row].index(min(x for x in removal_map[row] if x > 0))
    worst_board[row][col] = (worst_board[row][col] + 1) % board_size
    if worst_board[row][col] == 0:
        worst_board[row][col] = board_size
    removal_map[row][col] += 1
    # evaluate fitness and mark "done" if it's correct
    # if the row, col, and box are all correct now, we're done with this square
    if check_square(worst_board, row, col):
        for board in board_list:
            board[row][col] = worst_board[row][col]
    return board_list, removal_map


def crossover_operator(board_list, removal_map):
    """
    Performs crossover by pairing the worst board to the best board and working inwards. Crossover is done through the
    selection of a random row and the most mutated position and copying it from the better board to the worse board.
    :param board_list: the list of boards under analysis
    :param removal_map: the mapping which indicates which indices need to be updated
    :return: updated list of crossed-over boards
    """
    board_size = len(board_list[0])
    # determine the fitness of each board
    fitness_values = [fitness_function(board) for board in board_list]
    # we'll iterate until we've gotten everything
    while sum(fitness_values) > 0:
        least_fit_idx = fitness_values.index(max(fitness_values))
        most_fit_idx = fitness_values.index(min(x for x in fitness_values if x > 0))
        least_fit_idx = least_fit_idx if least_fit_idx != most_fit_idx else (least_fit_idx + 1) % board_size
        fitness_values[least_fit_idx] = 0
        fitness_values[most_fit_idx] = 0
        # select a position to crossover
        row = randint(0, board_size-1)
        col = removal_map[row].index(max(removal_map[row]))
        board_list[least_fit_idx][row][col] = board_list[most_fit_idx][row][col]
    return board_list


############################################### SUDOKU GENERATION ###############################################


def generate_sudoku(n):
    """
    Generates a solvable Sudoku board.
    In the interest of citation and academic honesty, the generation code is based on the information here:
    https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python
    :param n: the dimension of the interior boxes, i.e. the board is n^2 * n^2
    :return: a solvable Sudoku board matrix n^2 * n^2
    """
    side = n*n
    nums = sample(range(1, side + 1), side)  # random numbers
    board = [[nums[(n * (r % n) + r // n + c) % side] for c in range(side)] for r in range(side)]
    rows = [r for g in sample(range(n), n) for r in sample(range(g * n, (g + 1) * n), n)]
    cols = [c for g in sample(range(n), n) for c in sample(range(g * n, (g + 1) * n), n)]
    board = [[board[r][c] for c in cols] for r in rows]
    return board


def generate_removal_map(solution_board):
    """
    Removes numbers from the board and return both the updated board and a map of which indices were removed.
    In the interest of citation and academic honesty, the removal code is based on the information here:
    https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python
    :param solution_board: the generated non-altered board
    :return: (the altered board, the removal matrix where 1 indicates removed and 0 indicates unaltered)
    """
    board = deepcopy(solution_board)
    side_length = len(board)
    removal_map = [[0 for _ in range(side_length)] for _ in range(side_length)]
    squares = side_length * side_length
    empty = squares * 3//4
    for p in sample(range(squares), empty):
        board[p // side_length][p % side_length] = 0
        removal_map[p // side_length][p % side_length] = 1
    return board, removal_map


def pretty_print(board):
    """
    Prints a board prettily with zeros represented as empty squares.
    In the interest of citation and academic honesty, the removal code is based on the information here:
    https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python
    :param board: the board to be printed
    """
    side = len(board)
    base = int(side ** 0.5)
    expand_line = lambda line : line[0]+line[5:9].join([line[1:5]*(base-1)]*base)+line[9:13]
    line0 = expand_line("╔═══╤═══╦═══╗")
    line1 = expand_line("║ . │ . ║ . ║")
    line2 = expand_line("╟───┼───╫───╢")
    line3 = expand_line("╠═══╪═══╬═══╣")
    line4 = expand_line("╚═══╧═══╩═══╝")
    symbol = " 1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    nums = [[""] + [symbol[n] for n in row] for row in board]
    print(line0)
    for r in range(1, side + 1):
        print("".join(n + s for n, s in zip(nums[r - 1], line1.split("."))))
        print([line2, line3, line4][(r % side == 0) + (r % base == 0)])


#################################################################################################################


if __name__ == "__main__":
    seed(123)
    n = 3 # n is meant to be the dimension of the interior boxes, i.e. the board is n^2 * n^2
    population_size = 10
    # generate the sudoku board
    solved_board = generate_sudoku(n)
    # generate the puzzle board and the map of where we removed values
    puzzle_board, removed_map = generate_removal_map(solved_board)
    # create an array of this puzzle board to form the population
    puzzle_board_list = [deepcopy(puzzle_board) for _ in range(1, population_size)]

    pretty_print(puzzle_board)
    fitness_function(puzzle_board)
    # fig = plt.gcf()
    # fig.show()
    # fig.canvas.draw()
    time = 0
    best_fitness = -1
    while best_fitness != 0:
        # mutate the list
        puzzle_board_list, removed_map = mutation_operator(puzzle_board_list, removed_map)
        # crossover the list
        puzzle_board_list = crossover_operator(puzzle_board_list, removed_map)
        # plot the fitness value
        best_fitness = get_best_fitness(puzzle_board_list)
        print(str(time) + ": " + str(best_fitness))
        # plt.scatter(time, best_fitness)
        # plt.pause(0.01)
        # fig.canvas.draw()
        time += 1
    plt.ioff()
    plt.show()
    solution = get_best_board(puzzle_board_list)
    pretty_print(solved_board)
    pretty_print(solution)
