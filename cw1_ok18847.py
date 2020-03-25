import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

# Shape types
linear = "linear"
poly = "poly"
sin = "sin"


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
                return sys.argv[1], True

        return sys.argv[1], None

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
    list_xs = np.split(list_xs, num_segments)
    list_ys = np.split(list_ys, num_segments)

    # Get constants/coefficients and residuals
    sum_res = 0
    for i, xs in enumerate(list_xs):
        cs, res, y_hat, shape_type = least_squares_residual_type(xs, list_ys[i])
        print(shape_type)
        sum_res += res

        # Plot lines if specified
        if plot:
            plt.plot(xs, estimated_y(cs, xs, shape_type), 'r-')

    # Print residual sum of squares (RSS)
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

    :param xs: List/array-like of x values
    :param ys: List/array-like of y values
    :return: Matrix form A = [a, b, c, ...]
    """

    mean_cve = kfold_cross_val(5, xs, ys)

    ones = np.ones(xs.shape)
    xs_1 = np.column_stack((ones, xs))  # 1st degree (Linear)
    xs_2 = np.column_stack((xs_1, xs ** 2))  # 2nd degree (Quadratic) (Not evaluated)
    xs_3 = np.column_stack((xs_2, xs ** 3))  # 3rd degree (Cubic)
    xs_4 = np.column_stack((xs_3, xs ** 4))  # 4th degree (Not evaluated)
    xs_sin = np.column_stack((ones, np.sin(xs)))  # Sinusoidal

    # Constants/Coefficients
    cs_deg_1 = np.linalg.inv(xs_1.T.dot(xs_1)).dot(xs_1.T).dot(ys)
    cs_deg_2 = np.linalg.inv(xs_2.T.dot(xs_2)).dot(xs_2.T).dot(ys)
    cs_deg_3 = np.linalg.inv(xs_3.T.dot(xs_3)).dot(xs_3.T).dot(ys)
    cs_deg_4 = np.linalg.inv(xs_4.T.dot(xs_4)).dot(xs_4.T).dot(ys)
    cs_sin = np.linalg.inv(xs_sin.T.dot(xs_sin)).dot(xs_sin.T).dot(ys)

    # Hashmap where
    #   k = Cross validation error
    #   v = Tuple of constants/coefficients, residual, estimated_y and shape_type for xs
    hashmap = {
        mean_cve[0]: (cs_deg_1, residual(cs_deg_1, xs, ys, linear), estimated_y(cs_deg_1, xs, linear), linear),
        mean_cve[1]: (cs_deg_4, residual(cs_deg_4, xs, ys, poly), estimated_y(cs_deg_4, xs, poly), poly),
        mean_cve[2]: (cs_sin, residual(cs_sin, xs, ys, sin), estimated_y(cs_sin, xs, sin), sin)
    }

    # Get the min residual and its corresponding constants/coefficients
    min_cve = min(hashmap)
    return hashmap.get(min_cve)


# %% Define K-fold cross validation
def kfold_cross_val(k, xs, ys):
    # K-fold validation attributes
    kf = KFold(k, True)
    mean_cve = np.zeros(3)  # Array of cross validation errors

    # K-fold validation
    for train_index, test_index in kf.split(xs):
        xs_train, xs_test = xs[train_index], xs[test_index]
        ys_train, ys_test = ys[train_index], ys[test_index]

        ones = np.ones(xs_train.shape)  # Extend the first column with 1s
        xs_1 = np.column_stack((ones, xs_train))  # 1st degree (Linear)
        xs_2 = np.column_stack((xs_1, xs_train ** 2))  # 2nd degree (Quadratic) (Not evaluated)
        xs_3 = np.column_stack((xs_2, xs_train ** 3))  # 3rd degree (Cubic)
        xs_4 = np.column_stack((xs_3, xs_train ** 4))  # 4th degree (Not evaluated)
        xs_sin = np.column_stack((ones, np.sin(xs_train)))  # Sinusoidal

        # Constants/Coefficients
        cs_train_deg_1 = np.linalg.inv(xs_1.T.dot(xs_1)).dot(xs_1.T).dot(ys_train)
        cs_train_deg_2 = np.linalg.inv(xs_2.T.dot(xs_2)).dot(xs_2.T).dot(ys_train)
        cs_train_deg_3 = np.linalg.inv(xs_3.T.dot(xs_3)).dot(xs_3.T).dot(ys_train)
        cs_train_deg_4 = np.linalg.inv(xs_4.T.dot(xs_4)).dot(xs_4.T).dot(ys_train)
        cs_train_sin = np.linalg.inv(xs_sin.T.dot(xs_sin)).dot(xs_sin.T).dot(ys_train)

        # Cross validation error
        mean_cve[0] += residual(cs_train_deg_1, xs_test, ys_test, linear)
        mean_cve[1] += residual(cs_train_deg_4, xs_test, ys_test, poly)
        mean_cve[2] += residual(cs_train_sin, xs_test, ys_test, sin)

    # Calculate mean of all sum of cves
    for i in range(3):
        mean_cve[i] /= k

    return mean_cve


# %% Define residual
def residual(cs, xs, ys, shape_type):
    """
    Calculate residual sum of squares of 20 data points

    :param cs: Constants/coefficients of polynomials
    :param xs: List/array-like of x values
    :param ys: List/array-like of y values
    :param shape_type: String dictating what the equation type is
    :return: Residual error (int)
    """

    return np.sum((ys - estimated_y(cs, xs, shape_type)) ** 2)


# %% Define estimated y values (y_hat)
def estimated_y(cs, xs, shape_type):
    """
    Calculate estimated y values to plot

    :param cs: Constants/coefficients of polynomials
    :param xs: List/array-like of x values
    :param shape_type: String dictating what the equation type is
    :return: List/array-like of estimated y values
    """
    y_hat = 0

    # Calculated like:
    # y_hat = a + bsin(xs)
    if shape_type == "sin":
        y_hat = cs[0] + cs[1] * np.sin(xs)

    # Calculated like:
    # y_hat = a * xs^0 +
    #         b * xs^1 +
    #         ...
    else:
        for i in range(len(cs)):
            y_hat += cs[i] * xs ** i

    return y_hat


# %% Define main
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


# %% Run main
main("basic_1.csv")
main("basic_2.csv")
main("basic_3.csv")
main("basic_4.csv")
main("basic_5.csv")
main("adv_1.csv")
main("adv_2.csv")
main("adv_3.csv")
main("noise_1.csv")
main("noise_2.csv")
main("noise_3.csv")
# main()
