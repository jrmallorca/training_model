import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# %% Define functions
def main(f=None):
    """
    Main function run when executing file
    :param f: Filename of data file. If none inputted, assume
              input comes from 2nd arg when executed.
    :return: None
    """
    f_p, plot = get_file_and_plot(f)
    if f_p is None:
        sys.exit(0)

    xs, ys = load_points_from_file(f_p)
    view_data_segments(xs, ys, plot)


def get_file_and_plot(f):
    """
    Checks if file specified when running in Python Console or
    when compiled
    :param f: Filename of data file. If none inputted, assume
              input comes from 2nd arg when executed.
    :return: Path to file or None
    """
    if f:  # Check if file specified when calling main func
        return "train_data/{}".format(f), True

    elif len(sys.argv) > 1:  # Check if file specified from user input
        if sys.argv[1] == "--mode=client":  # Check if run from Python Console
            print("ERROR: Please specify the file to get data from")
            return None, None

        elif len(sys.argv) > 2:  # Check if user wants data plotted
            if sys.argv[2] == "--plot":
                return "train_data/{}".format(sys.argv[1]), True

        return "train_data/{}".format(sys.argv[1]), None

    else:  # Check if filename unspecified anywhere
        print("ERROR: Please specify the file to get data from")
        return None, None


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

    # Get constants/coefficients and total residual error
    a, b = least_squares(xs, ys)
    print('RSS = {}'.format(total_residual(a, b, xs, ys)))

    # Plot line if specified
    if plot:
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = a + b * xmin, a + b * xmax

        plt.plot([xmin, xmax], [ymin, ymax], 'r-')
        plt.show()


def least_squares(xs, ys):
    """
    Calculates a and b
    (linear: y = a + b*x_i)

    :param xs: ndarray of x values
    :param ys: ndarray of y values
    :return: Matrix form A = [a, b]
    """
    # Extend the first column with 1s
    ones = np.ones(xs.shape)
    x_e = np.column_stack((ones, xs))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(ys)
    return v


def total_residual(a, b, xs, ys):
    """
    Calculate residual sum of squares

    :param a: y-intercept
    :param b: Gradient of the slope
    :param xs: ndarray of x values
    :param ys: ndarray of y values
    :return: Total residual error (int)
    """
    y_hat = a + b * xs
    return np.sum((ys - y_hat) ** 2)


# %% Run main
main("basic_1.csv")
