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
        cs, res = least_squares(xs, list_ys[i])
        sum_res += res

        # Plot lines if specified
        if plot:
            y_plot = 0
            for j in range(len(cs)):
                y_plot += cs[j] * xs**j

            plt.plot(xs, y_plot, 'r-')

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

    # 1st degree (Linear)
    xs_1 = np.column_stack((ones, xs))
    deg_1 = np.linalg.inv(xs_1.T.dot(xs_1)).dot(xs_1.T).dot(ys)

    # 2nd degree (Quadratic)
    xs_2 = np.column_stack((xs_1, xs**2))
    deg_2 = np.linalg.inv(xs_2.T.dot(xs_2)).dot(xs_2.T).dot(ys)

    # 3rd degree (Cubic)
    xs_3 = np.column_stack((xs_2, xs**3))
    deg_3 = np.linalg.inv(xs_3.T.dot(xs_3)).dot(xs_3.T).dot(ys)

    # Dictionary (hash map) with residual as key and value as the matrix
    dict = {residual(deg_1, xs, ys): deg_1,
            residual(deg_2, xs, ys): deg_2,
            residual(deg_3, xs, ys): deg_3}

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
main("noise_3.csv", True)
