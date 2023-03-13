# Sparse neural network regression with variable selection (SBNN)

This is a neural network regression method. 
The method uses a neural network structure with a hidden layer and B-spline functions as node activation.
A penalization scheme is applied to avoid overfitting and to select significant variables.
An algorithm is used, which adds nodes to neural network whenever a tuning parameter increases.

### Data and goal
Consider a dataset $\{(y_i, x_i) \}_{i=1}^n$ gererated from the regression model

$$
y_i = f(x_i) + \varepsilon_i, \quad i = 1, \ldots, p
$$

where $y_i$ is response and $x_i \in \mathbb{R}^p$ is predictors.
The goal is to estimate the target function $f$ based on the observed data.

### Neural network
A neural network with a hidden layer is given by

$$
\mathsf{f}_{\theta}(x) = \beta_0 + \sum_{m=1}^M \beta_m \sigma(b_m + w_m^{\top} x), \quad x \in \mathbb{R}^p,
$$

where \theta is the parameter vector including all parameters in \mathsf{f}_{\theta} .
The $\sigma$ denotes B-spline activation function.

### Objective function
The objective function is the sum of loss function and penalty function, which is given by

$$
R^{\lambda}(\theta) = \frac{1}{2N} \left(y_i - \mathsf{f}_{\theta}(x_i) \right)^2 + \lambda_1 \left|\beta\right|_1 + \lambda_2 \sum_{j=1}^p \left| w^j \right|_2
$$

where $w^j = (w_{1j}, \ldots, w_{mj})$ is the weight vector associated to the $j$ th input.


The $\lambda = (\lambda_1, \lambda_2)$ is the tuning parameter vector. 

- $\lambda_1$ penalizes the effect of nodes to output and enables neural network remove unnecessary nodes as $\lambda_1$ increases.
- $\lambda_2$ penalizes the group effects of each input to nodes and remove unnecessary inputs as $\lambda_2$ increases.

### Estimator
The sparse B-spline neural network estimator is given by 

$$
\hat f = \mathsf{f}_{\hat \theta}, \quad \hat \theta = \text{argmin}_{\theta} R^{\lambda}(\theta).
$$

### Figures

- B-spline activation function
![BAF-1](https://user-images.githubusercontent.com/84615460/224712089-2af5f480-025a-4292-8cbb-fe4bda317608.png)

- penalization scheme
![penalty-1](https://user-images.githubusercontent.com/84615460/224712132-1d2bf953-1794-418a-bd80-cc83b5d215bb.png)
