### libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

### FUNCTIONS

##Basic linear regression by the least-squares method
def linear_regression(xs, ys):
    column_of_ones = np.vstack((np.ones(xs.shape[0]), xs)).T
    parameters = np.linalg.inv(column_of_ones.T @ column_of_ones) @ column_of_ones.T @ ys
    
    return parameters

##Piecewise linear regression fit of data
def piecewise_linear_fit(x, y, breakpoints):
    models = []  # Linear fit for each segment
    segments_x = []
    segments_y = []
    
    breakpoints = [0] + breakpoints + [len(x)]
    
    for i in range(1, len(breakpoints)):
        start = breakpoints[i-1]
        end = breakpoints[i]
        x_segment = x[start:end]
        y_segment = y[start:end]
        
        a, b = linear_regression(x_segment, y_segment)
        models.append((a, b))
        
        segments_x.append(x_segment)
        segments_y.append(y_segment)
    
    return models, segments_x, segments_y

##Plotting my piecewise lienar regression fit of data
def plot_my_piecewise_regression(x, y, breakpoints):
    models, segments_x, _ = piecewise_linear_fit(x, y, breakpoints)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Data points')
    
    breakpoints = [0] + breakpoints + [len(x)]
    
    #plot line for each segment
    for i in range(len(models)):
        a, b = models[i]
        start = breakpoints[i]
        end = breakpoints[i + 1]
        x_segment = x[start:end]
        y_segment = a + b * x_segment
        
        plt.plot(x_segment, y_segment, label=f'Segment {i+1}', linestyle='--')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('My model for piece-wise linear regression of data')
    plt.legend()
    plt.show()

##Piecewise linear regression by numpy function
def piecewise_linear_by_numpy(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0, x >= x0], [lambda x: k1*x + y0 - k1*x0, lambda x: k2*x + y0 - k2*x0])

##Compare both regressions
def compare_regressions(x, y, breakpoints):
    
    #my fit
    models, segments_x, segments_y = piecewise_linear_fit(x, y, breakpoints)
    
    #numpy fit
    p0 = [x[breakpoints[0]], y[breakpoints[0]], 1, 1]  # Initial guess for the parameters
    p, _ = optimize.curve_fit(piecewise_linear_by_numpy, x, y, p0)
    xd = np.linspace(min(x), max(x), 100)
    
    #plotting both fits on the same plot
    plt.scatter(x, y, color='blue', label='Data points')

    #plotting my fit on the plot
    for i in range(len(models)):
        a, b = models[i]
        x_range = np.linspace(segments_x[i].min(), segments_x[i].max(), 100)
        y_range = a + b * x_range
        plt.plot(x_range, y_range, label=f'Segment {i+1} - Custom', linestyle='--')

    #plotting the numpy fit on the plot
    plt.plot(xd, piecewise_linear_by_numpy(xd, *p), label='Piecewise linear fit - Numpy', color='red')
    plt.title('Comparison of both models for piece-wise linear regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


##Statistical validation of both fits

##function to calculate AIC
def calculate_aic(n, residual_sum_of_squares, k):
    return 2 * k + n * np.log(residual_sum_of_squares / n)

##function to calculate BIC
def calculate_bic(n, residual_sum_of_squares, k):
    return k * np.log(n) + n * np.log(residual_sum_of_squares / n)

##function to validate a regression model
def validate_regression(y, y_fit, k):
    n = len(y)
    
    # RMSE -- Root Mean Squared Error 
    rmse = np.sqrt(mean_squared_error(y, y_fit))
    
    # MAE -- Mean Absolute Error 
    mae = mean_absolute_error(y, y_fit)
    
    # R-squared
    r_squared = r2_score(y, y_fit)
    
    # Residual sum of squares
    residual_sum_of_squares = np.sum((y - y_fit) ** 2)
    
    # AIC -- Akaike Information Criterion
    aic = calculate_aic(n, residual_sum_of_squares, k)
    
    # BIC -- Bayesian Information Criterion
    bic = calculate_bic(n, residual_sum_of_squares, k)
    
    return {
        "RMSE": rmse, #lower is better
        "MAE": mae, #lower is better
        "R-squared": r_squared, #higher is better
        "AIC": aic, #lower is better
        "BIC": bic #lower is better
    }

##validation for custom piecewise regression
def validate_my_fit(x, y, breakpoints):
    models, segments_x, _ = piecewise_linear_fit(x, y, breakpoints)
    y_pred = np.zeros_like(y)
    
    breakpoints = [0] + breakpoints + [len(x)]  # Adjust breakpoints to include start and end points
    
    for i, (a, b) in enumerate(models):
        start = breakpoints[i]
        end = breakpoints[i + 1]
        y_pred[start:end] = a + b * x[start:end]
    
    validation_metrics = validate_regression(y, y_pred, len(models) * 2)  # 2 parameters per segment (a, b)
    
    print("Custom Piecewise Regression Validation Metrics:")
    
    for key, value in validation_metrics.items():
        print(f"{key}: {value}")
        
    return validation_metrics

##validation for numpy piecewise regression
def validate_numpy_fit(x, y, p):
    y_pred = piecewise_linear_by_numpy(x, *p)
    validation_metrics = validate_regression(y, y_pred, len(p))
    
    print("Numpy Piecewise Regression Validation Metrics:")
    
    for key, value in validation_metrics.items():
        print(f"{key}: {value}")
        
    return validation_metrics


### USE ON DATA ###

#Read the data from csv (or xls)
data = pd.read_csv("data.csv", delimiter=";")
#or data = pd.read_excel("data.xls")

x = data["x"].values
y = data["y"].values

#Visualization of the given data
plt.scatter(x, y, label='Data', color='b', marker='o')
plt.title('Visualization of data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Identifying the breakpoints
my_breakpoints = list(int(i) for i in input("Enter breakpoints: ").split())

#Plot the custom regression
plot_my_piecewise_regression(x, y, my_breakpoints)

#Compare both regressions
compare_regressions(x, y, my_breakpoints)

#Validate both models
p0 = [x[my_breakpoints[0]], y[my_breakpoints[0]], 1, 1]
p, _ = optimize.curve_fit(piecewise_linear_by_numpy, x, y, p0) #p is the parameter from numpy piecewise regression

custom_metrics = validate_my_fit(x, y, my_breakpoints)
numpy_metrics = validate_numpy_fit(x, y, p)