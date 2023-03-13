# Sparse neural network regression with variable selection (SBNN)

### Neural network
A neural network with a hidden layer is given by

$$
\mathsf{f}_{\theta}(x) = \beta_0 + \sum_{m=1}^M \beta_m \sigma(b_m + w_m^{\top} x), \quad x \in \mathbb{R}^p,
$$

where \theta is the parameter vector including all parameters in \mathsf{f}_{\theta}.

### Objective function
The objective function is the sum of loss function and penalty function, which is given by

$$
R^{\lambda}(\theta) = \frac{1}{2N} \left(y_i - \mathsf{f}_{\theta}(x_i) \right)^2 + \lambda_1 \norm{\beta} + \lambda_2 \sum_{j=1}^p \norm{w^j}
$$
where $w_j = (w_{m1}, \ldots, w_{mp})$.
