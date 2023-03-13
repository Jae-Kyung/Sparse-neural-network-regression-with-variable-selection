# Sparse neural network regression with variable selection (SBNN)

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

where \theta is the parameter vector including all parameters in \mathsf{f}_{\theta}.

### Objective function
The objective function is the sum of loss function and penalty function, which is given by

$$
R^{\lambda}(\theta) = \frac{1}{2N} \left(y_i - \mathsf{f}_{\theta}(x_i) \right)^2 + \lambda_1 \left|\beta\right|_1 + \lambda_2 \sum_{j=1}^p \left| w^j \right|_2
$$

where $w_j = (w_{1j}, \ldots, w_{mj})$.
