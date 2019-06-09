from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


def create_dataset(num_of_samples, variance, step=2, correlation=False):
    """
    Generates a random dataset

    INPUTS
        num_of_samples: number of data points in the dataset; an int value
        variance: the divergence of the dataset
        step: degree of increment or decrement in Y values
        correlation: the relation between X values and Y values

    OUPUT
        returns two arrays - Xs and their corresponding Ys
    """

    val = 1
    ys = []
    for i in range(num_of_samples):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    '''
    Calculates the gradient and y-intercept of the line of best fit using
    the Least Square Method

    INPUTS
        Xs and Ys of the dataset

    OUTPUT
        returns the value of m and b
    '''

    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs)**2) - mean(xs**2)))

    b = mean(ys) - m * mean(xs)
    return m, b


def squared_error(ys_orig, ys_line):
    '''
    Calculates the squared error (SE) of the regression line

    INPUTS
        ys_orig: the real labels of Xs
        ys_line: predicted labels of Xs

    OUTPUT
        returns the squared error of the regression line
    '''
    return sum((ys_orig - ys_line)**2)


def coefficient_of_determination(ys_orig, ys_line):
    '''
    Calculates R squared value

    INPUTS
        The true and predicted values of Y

    OUTPUT
        returns the R squared value of the model
    '''

    y_mean_line = [mean(ys_orig)for y in ys_orig]

    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)

    return 1 - (squared_error_regr / squared_error_y_mean)


xs, ys = create_dataset(40, 40, 2, correlation='pos')

m, b = best_fit_slope_and_intercept(xs, ys)

# print(m, b)

regression_line = [(m * x) + b for x in xs]

# predicting what 10 will yield
predict_x = 10
predict_y = (m * predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys, color='b')
plt.scatter(predict_x, predict_y, color='g')  # printing the prediction
plt.plot(xs, regression_line, color='k')
plt.show()
