import sys

import utilities as ut


# %% Define main
def main(f=None):
    """
    Main function run when executing file
    :param f: Filename of data file. If none inputted, assume
              input comes from 2nd arg when executed.
    :return: None
    """
    f_p, plot = ut.get_file_and_plot(f)
    if f_p is None:
        sys.exit(0)

    xs, ys = ut.load_points_from_file(f_p)
    ut.view_data_segments(xs, ys, plot)


# %% Run main
main()