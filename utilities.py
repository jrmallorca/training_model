import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# %% Define functions
def main(f=None, plot=None):
    """
    Main function run when executing file
    :param f: Filename of data file. If none inputted, assume
              input comes from 2nd arg when executed.
    :param plot: Boolean value to indicate if we should show
                 plotted lines
    :return: None
    """
    f_p = get_file(f)
    if f_p is None:
        sys.exit(0)

    xs, ys = load_points_from_file(f_p)
    view_data_segments(xs, ys, plot)


def get_file(f):
    """
    Checks if file specified when running in Python Console or
    when compiled
    :param f: Filename of data file. If none inputted, assume
              input comes from 2nd arg when executed.
    :return: Path to file or None
    """
    if f:  # Check if file specified when calling main func
        return "train_data/{}".format(f)

    elif len(sys.argv) > 1:  # Check if file specified from user input
        if sys.argv[1] == "--mode=client":  # Check if run from Python Console
            print("ERROR: Please specify the file to get data from")
            return None

        elif len(sys.argv) > 2:  # Check if user wants data plotted
            if sys.argv[2] == "--plot":
                return "train_data/{}".format(sys.argv[1])

        return "train_data/{}".format(sys.argv[1])

    else:  # Check if filename unspecified anywhere
        print("ERROR: Please specify the file to get data from")
        return None


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys, plot):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
        plot : Boolean specifying whether user wants data plotted
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)

    # Convert ndarray into list of ndarrays of 20 x-values
    xs = np.split(xs, len(xs) / 20)
    ys = np.split(ys, len(ys) / 20)

    # Get constants/coefficients and total residual error
    list_a, list_b = [], []
    for i, x in enumerate(xs):
        a, b = least_squares(x, ys[i])
        list_a.append(a)
        list_b.append(b)

        # Plot lines if specified
        if plot:
            xmin, xmax = xs[i].min(), xs[i].max()
            ymin, ymax = a + b * xmin, a + b * xmax

            plt.plot([xmin, xmax], [ymin, ymax], 'r-')

    print('RSS = {}'.format(total_residual(list_a, list_b, xs, ys)))

    # Show plotted line(s) if specified
    if plot:
        plt.show()


def least_squares(xs, ys):
    """
    Calculates a and b
    (linear: y = a + b*x_i)

    :param xs: ndarray of x values
    :param ys: ndarray of y values
    :return: Matrix form A = [a, b]
    """
    ones = np.ones(xs.shape)  # Extend the first column with 1s
    x_e = np.column_stack((ones, xs))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(ys)

    return v


def total_residual(list_a, list_b, list_xs, list_ys):
    """
    Calculate residual sum of squares

    :param list_a: y-intercepts
    :param list_b: Slope gradients
    :param list_xs: List of ndarrays of x values
    :param list_ys: List of ndarrays of y values
    :return: Total residual error (int)
    """
    sum = 0
    for i, a in enumerate(list_a):
        # ∑_𝑖 (𝑦̂_𝑖 − 𝑦_𝑖)^2 where 𝑦̂_𝑖 = 𝑎 + 𝑏*𝑥_𝑖
        sum += np.sum((list_ys[i] - (a + list_b[i] * list_xs[i])) ** 2)

    return sum


# %% Run main
main("basic_3.csv", True)
