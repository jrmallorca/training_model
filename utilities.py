import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# %% Define main
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


# %% Define get file
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


# %% Define loading points from file
def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


# %% Define viewing data
def view_data_segments(xs, ys, plot):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
        plot : Boolean specifying whether user wants data plotted
    Returns:
        None
    """
    # Preparation
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

    # Get constants/coefficients and residuals
    sum_res = 0
    for i, x in enumerate(xs):
        cs, res = least_squares(x, ys[i])

        sum_res += res

        # Plot lines if specified
        if plot:
            x_min, x_max = xs[i].min(), xs[i].max()

            y_min, y_max = 0, 0
            for i in range(len(cs)):
                y_min += cs[i] * x_min**i
                y_max += cs[i] * x_max**i

            plt.plot([x_min, x_max], [y_min, y_max], 'r-')

    # Print total residual
    print('RSS = {}'.format(sum_res))

    # Show plotted line(s) if specified
    if plot:
        plt.show()


# %% Define least squares solution
def least_squares(xs, ys):
    """
    Calculates a and b
    (linear: y = a + b*x_i)
    (quadratic: y = a + b*x_i + c*x_i^2)
    etc...

    :param xs: ndarray of x values
    :param ys: ndarray of y values
    :return: Matrix form A = [a, b, c, ...]
    """
    ones = np.ones(xs.shape)  # Extend the first column with 1s

    # Linear
    xs_l = np.column_stack((ones, xs))
    l = np.linalg.inv(xs_l.T.dot(xs_l)).dot(xs_l.T).dot(ys)

    # Quadratic
    xs_q = np.column_stack((xs_l, xs**2))
    q = np.linalg.inv(xs_q.T.dot(xs_q)).dot(xs_q.T).dot(ys)

    # Dictionary (hash map) with residual as key and value as the matrix
    dict = {residual(l, xs, ys): l,
            residual(q, xs, ys): q}

    # Get the min residual and its corresponding constants/coefficients
    min_res = min(dict)
    return dict.get(min_res), min_res


# %% Define residual of 20 points
def residual(cs, xs, ys):
    """
    Calculate residual sum of squares of 20 data points
    ∑_i (𝑦̂_i − y_i)^2 where 𝑦̂_i = a + b*x_i + c*x_i^2 + ...

    :param cs: Constants/coefficients of polynomials
    :param xs: ndarrays of x values
    :param ys: ndarrays of y values
    :return: Total residual error (int)
    """
    # Calculated like:
    # y_hat = a * xs^0 +
    #         b * xs^1 +
    #         ...
    #         c * xs^n
    y_hat = 0
    for i in range(len(cs)):
        y_hat += cs[i] * xs**i

    return np.sum((ys - y_hat) ** 2)


# %% Run main
main("basic_1.csv", True)
