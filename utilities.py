import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# %% Define get file
def get_file_and_plot(f):
    """
    Checks if file specified when running in Python Console or
    when compiled
    :param f: Filename of data file. If none inputted, assume
              input comes from 2nd arg when executed.
    :return: Path to file or None, and Boolean to show graph or not
    """
    if f:  # Check if file specified when calling main func
        return "train_data/{}".format(f), True

    elif len(sys.argv) > 1:  # Check if file specified from user input
        if sys.argv[1] == "--mode=client":  # Check if run from Python Console
            print("ERROR: Please specify the file to get data from")
            return None, None

        if len(sys.argv) > 2:  # Check if user wants data plotted
            if sys.argv[2] == "--plot":
                return "train_data/{}".format(sys.argv[1]), True

        return "train_data/{}".format(sys.argv[1]), None

    else:  # Check if filename unspecified anywhere
        print("ERROR: Please specify the file to get data from")
        return None, None


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
def view_data_segments(list_xs, list_ys, plot):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        list_xs : List/array-like of x co-ordinates.
        list_ys : List/array-like of y co-ordinates.
        plot : Boolean specifying whether user wants data plotted
    Returns:
        None
    """
    # Preparation
    assert len(list_xs) == len(list_ys)
    assert len(list_xs) % 20 == 0
    len_data = len(list_xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(list_xs, list_ys, c=colour)

    # Convert ndarray into list of ndarrays of 20 x-values
    list_xs = np.split(list_xs, len(list_xs) / 20)
    list_ys = np.split(list_ys, len(list_ys) / 20)

    # Get constants/coefficients and residuals
    sum_res = 0
    for i, xs in enumerate(list_xs):
        cs, shape_type, res = least_squares_residual_type(xs, list_ys[i])
        sum_res += res

        # Plot lines if specified
        if plot:
            y_plot = 0
            if shape_type == "sin":
                y_plot = cs[0] + cs[1] * np.sin(xs)
            else:
                for j in range(len(cs)):
                    y_plot += cs[j] * xs**j

            plt.plot(xs, y_plot, 'r-')

    # Print total residual
    print('RSS = {}'.format(sum_res))

    # Show plotted line(s) if specified
    if plot:
        plt.show()


# %% Define least squares solution
def least_squares_residual_type(xs, ys):
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

    # 1st degree (Linear)
    xs_1 = np.column_stack((ones, xs))
    deg_1 = np.linalg.inv(xs_1.T.dot(xs_1)).dot(xs_1.T).dot(ys)

    # 2nd degree (Quadratic)
    xs_2 = np.column_stack((xs_1, xs**2))
    deg_2 = np.linalg.inv(xs_2.T.dot(xs_2)).dot(xs_2.T).dot(ys)

    # 3rd degree (Cubic)
    xs_3 = np.column_stack((xs_2, xs**3))
    deg_3 = np.linalg.inv(xs_3.T.dot(xs_3)).dot(xs_3.T).dot(ys)

    # Sinusoidal
    xs_sin = np.column_stack((ones, np.sin(xs)))
    sin = np.linalg.inv(xs_sin.T.dot(xs_sin)).dot(xs_sin.T).dot(ys)

    # Dictionary (hash map) with residual as key and value as the matrix
    dict = {residual(deg_1, xs, ys, "poly"): (deg_1, "poly"),
            residual(deg_2, xs, ys, "poly"): (deg_2, "poly"),
            residual(deg_3, xs, ys, "poly"): (deg_3, "poly"),
            residual(sin, xs, ys, "sin"): (sin, "sin")}

    # Get the min residual and its corresponding constants/coefficients
    min_res = min(dict)
    cs, shape_type = dict.get(min_res)
    return cs, shape_type, min_res


# %% Define residual of 20 points
def residual(cs, xs, ys, shape_type):
    """
    Calculate residual sum of squares of 20 data points
    ∑_i (𝑦̂_i − y_i)^2 where 𝑦̂_i = a + b*x_i + c*x_i^2 + ...

    :param cs: Constants/coefficients of polynomials
    :param xs: ndarrays of x values
    :param ys: ndarrays of y values
    :param shape_type: String dictating what the equation type is
    :return: Residual error (int)
    """
    y_hat = 0

    # Calculated like:
    # y_hat = a + bsin(xs)
    if shape_type == "sin":
        y_hat = cs[0] + cs[1] * np.sin(xs)

    # Calculated like:
    # y_hat = a * xs^0 +
    #         b * xs^1 +
    #
    else:
        for i in range(len(cs)):
            y_hat += cs[i] * xs**i

    return np.sum((ys - y_hat) ** 2)
