"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    rng=np.random.default_rng(seed)
    indices=np.arange(len(X))
    rng.shuffle(indices)
    X=X[indices]
    y=y[indices]
    return X,y

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    n=X.shape[0]
    n_train=int(n*train_frac)
    n_val=int(n*val_frac)
    n_test=n-n_train-n_val

    X_train=X[:n_train,:]
    y_train=y[:n_train]
    
    X_val=X[n_train:n_train+n_val,:]
    y_val=y[n_train:n_train+n_val]

    X_test=X[n-n_test:,:]
    y_test=y[n-n_test:]
    return X_train, y_train, X_val, y_val, X_test, y_test

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    std[std==0]=1
    return mean,std

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    X_scaled=(X-mean)/std
    return X_scaled

# Step 5 - add_bias_column
def add_bias_column(X):
    n,d=X.shape
    new_col=np.ones((n,1))
    X=np.append(new_col,X,axis=1)
    return X

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    X_scaled=(X-mean)/std
    X_scaled=add_bias_column(X_scaled)
    return X_scaled

# Step 7 - predict_linear
def predict_linear(X, weights):
    y_hat=X@weights
    return y_hat

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    residual=y_true-y_pred 
    mean_squared_residual=residual**2/y_true.shape
    mse=np.sum(mean_squared_residual)
    return mse

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    n=y_true.shape[0]
    return (2/n)*(X.T@(y_pred-y_true))

# Step 10 - normal_equation
def normal_equation(X, y):
    weights=np.linalg.solve(X.T@X,X.T@y)
    return weights

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(0,0.01,n_features)

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    y_hat=predict_linear(X,weights)
    weights=weights-(lr*mse_gradient(X,y,y_hat))
    return weights

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    y_train_pred=predict_linear(X_train,weights)
    y_val_pred=predict_linear(X_val,weights)
    train_loss=mse_loss(y_train,y_train_pred)
    val_loss=mse_loss(y_val,y_val_pred)
    return (train_loss,val_loss)

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    if(val_loss<best_val_loss):
        return (val_loss,0,weights.copy(),False)
    else:
        return (best_val_loss,wait+1,best_weights,(wait+1)>=patience)

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    weights=initialize_weights(n_features,seed)
    return {
        "weights":weights.copy(),
        "best_weights":weights.copy(),
        "best_val_loss":np.inf,
        "wait":0,
        "train_losses":[],
        "val_losses":[],
        "stopped":False
    }

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    state["weights"]=gd_step(X_train,y_train,state["weights"],lr)
    train_loss,val_loss=epoch_train_val_losses(X_train,y_train,X_val,y_val,state["weights"])
    state["train_losses"].append(train_loss)
    state["val_losses"].append(val_loss)
    state["best_val_loss"],state["wait"],state["best_weights"],state["stopped"]=update_early_stop_state(
        val_loss,state["best_val_loss"],state["wait"],state["weights"],state["best_weights"],patience
    ) 
    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    state=init_training_state(X_train.shape[1],seed)
    for i in range(epochs):
        if(state["stopped"]):
            break
        state=run_one_epoch(state,X_train,y_train,X_val,y_val,lr,patience)
    return (state["best_weights"],state["train_losses"],state["val_losses"])

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

