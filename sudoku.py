#!/usr/bin/env python3.7

from copy import deepcopy
from random import sample, seed
from statistics import mean


############################################### GENETIC ALGORITHM ###############################################


def fitness_function(board):
    """
    Determines the fitness value (i.e., how close it is to a final solution) of the given board. This is done via
    averaging all of the dimensions of the board that need to add up to a final number mean(range(1,n**2+1))
    :param board: the board whose fitness is being evaluated
    :return: float value representing the fitness of the given board
    """
    side_length = len(board)
    base = int(side_length ** 0.5)
    # sum each row
    row_sum = sum([sum(row) for row in board])
    # sum each column
    column_sum = sum([sum(column) for column in zip(*board)])
    # sum each interior box
    box_sum = 0
    for row_idx in range(0, base):
        for col_idx in range(0, base):
            box = [row[row_idx*base:(row_idx+1)*base] for row in board[col_idx*base:(col_idx+1)*base]]
            box_sum += sum(sum(box, []))
    # find the average value to determine fitness
    board_fitness = (row_sum + column_sum + box_sum) / (side_length * 3)
    # determine and return the variation from the ideal fitness value
    ideal_fitness = mean(range(1,n**2+1)) * side_length
    return abs(ideal_fitness - board_fitness)


def mutation_operator(board, removal_map):
    pass


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
    seed(724) # DEVELOPER USE
    n = 3  # n is meant to be the dimension of the interior boxes, i.e. the board is n^2 * n^2
    solved_board = generate_sudoku(n)
    pretty_print(solved_board)
    puzzle_board, removed_map = generate_removal_map(solved_board)
    print()
    pretty_print(puzzle_board)
    print()
    mutation_operator(puzzle_board)