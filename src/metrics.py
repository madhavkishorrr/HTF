import numpy as np


def r_square(actual_values,predicted_values):
    # Calculate the mean actual and predicted values row-wise
    mean_actual = np.mean(actual_values, axis=1)
    mean_predicted = np.mean(predicted_values, axis=1)

    # Calculate the sum of squared differences between actual and predicted values row-wise
    ss_tot = np.sum(np.square(actual_values - mean_actual[:, np.newaxis]), axis=1)
    ss_res = np.sum(np.square(predicted_values - actual_values), axis=1)

    # Calculate R-squared row-wise
    rsquared = 1 - (ss_res / ss_tot)

    return rsquared

def mse_rmse(actual_values,predicted_values):
    # Calculate the squared errors row-wise
    squared_errors = np.square(actual_values - predicted_values)
    rowwise_squared_errors = np.sum(squared_errors, axis=1)
    # Calculate the RMSE row-wise
    rowwise_mse = (rowwise_squared_errors / actual_values.shape[1])
    rowwise_rmse = np.sqrt(rowwise_mse)
    return rowwise_mse,rowwise_rmse

def mae_mape(actual_values,predicted_values):
    # Calculate the absolute differences between actual and predicted values row-wise
    abs_diff = np.abs(predicted_values - actual_values)
    # Calculate MAE row-wise
    mae = np.mean(abs_diff, axis=1)
    # Calculate MAPE row-wise
    av  = actual_values.replace(0,0.01)
    av = av.fillna(0.001)
    mape = np.mean(abs_diff /av , axis=1) * 100

    return mae,mape

