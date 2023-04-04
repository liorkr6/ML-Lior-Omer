###### Your ID ######
# ID1: 209039056
# ID2: 208637124
#####################

# imports 
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    X = (X - np.mean(X, axis=0)) / (np.amax(X, axis=0) - np.amin(X, axis=0))
    y = (y - np.mean(y, axis=0)) / (np.amax(y, axis=0) - np.amin(y, axis=0))
    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    m = X.shape[0]
    col_of_ones = np.ones((m, 1))
    X = np.column_stack((col_of_ones, X))
    return X


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    m = y.size
    prediction = np.dot(X, theta)
    J = np.sum(np.square(prediction - y)) / (2 * m)
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    m = y.size
    for i in range(num_iters):
        prediction = np.dot(X, theta)
        mistake_val = prediction - y
        gradiant = np.dot(X.T, mistake_val)
        theta -= (alpha / m) * gradiant
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    pinv_x = np.matmul((np.linalg.inv(np.matmul(X.T, X))), X.T)
    pinv_theta = np.matmul(pinv_x, y)
    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    m, n = X.shape
    for i in range(num_iters):
        prediction = np.dot(X, theta)
        mistake_val = prediction - y
        gradient = np.dot(X.T, mistake_val)
        theta -= ((alpha / m) * gradient)
        J_history.append(compute_cost(X, y, theta))
        if i > 0 and J_history[i - 1] - J_history[i] < 1e-8:
            break
    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}
    start_theta = np.random.random(size=X_train.shape[1])
    for alpha in alphas:
        best_theta_efficient, _ = efficient_gradient_descent(X_train, y_train, start_theta, alpha, iterations)
        validation_loss = compute_cost(X_val, y_val, best_theta_efficient)
        alpha_dict[alpha] = validation_loss
    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    X_train, X_val = apply_bias_trick(X_train), apply_bias_trick(X_val)
    n = X_train.shape[1]
    for i in range(5):
        lowest_cost = np.inf
        first_theta = np.random.random(size=len(selected_features) + 1)
        for j in range(n):
            if j not in selected_features:
                columns = selected_features + [j]
                theta, _ = efficient_gradient_descent(X_train[:, columns], y_train, first_theta, best_alpha, iterations)
                cost_val = compute_cost(X_val[:, columns], y_val, theta)
            else:
                continue
            if cost_val < lowest_cost:
                lowest_cost = cost_val
                feature_index = j
        selected_features.append(feature_index)
    return [x - 1 for x in selected_features]


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """
    df_poly = df.copy()
    for i, i_column in enumerate(df.columns):
        for j, j_column in enumerate(df.columns[i:], start=i):
            if i == j:
                new_col = i_column + '^2'
            else:
                new_col = i_column + '*' + j_column
            df_poly[new_col] = df[i_column] * df[j_column]
    return df_poly
