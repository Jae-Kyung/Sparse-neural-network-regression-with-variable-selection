#  R code for sparse B-spline neural network regression
#  SBNN method is implemented
#  input
#     response = n x 1 response vector
#     predictors = n x (p + 1) predictor matrix whose first column contains ones,
#                  where p denotes the number of predictors
#  output
#     slm = slm list
# for implementation

# main function
SBNN = function(response,
                predictors,
                try_number_nodes,
                initial = FALSE,
                alpha = NA,
                beta,
                gamma,
                lambdas = 1e-10,
                lambdag = 1,
                step_size = 1,
                max_iterations = 1000,
                epsilon = 1e-4,
                threshold_g = 1e-03,
                threshold = 1e-20,
                verbose = FALSE)
{
   fslm = list()
   # sizes
   sample_size = length(response)
   number_predictors = ncol(predictors)
   # initialization planes, activated_planes and activated_planes_derivative
   planes = matrix(0, sample_size, try_number_nodes)
   activated_planes = matrix(0, sample_size, try_number_nodes)
   activated_planes_derivative = matrix(0, sample_size, try_number_nodes)
   if (!initial)
   {
      alpha = 0
      beta = c()
      gamma = matrix(0, number_predictors, 0)
      fitted_values = rep(0, sample_size)
   }
   residuals = response - fitted_values
   store_rss = Inf
   # for initials
   keep_residuals = residuals
   keep_beta = beta
   keep_alpha = alpha
   keep_gamma = gamma
   # prepare for regularizations
   number_lambdas = length(lambdas)
   number_lambdag = length(lambdag)
   bic_mat = matrix(0, number_lambdas, number_lambdag)
   aic_mat = matrix(0, number_lambdas, number_lambdag)
   mix_mat = matrix(0, number_lambdas, number_lambdag)
   iter_mat = matrix(0, number_lambdas, number_lambdag)
   number_active_node_mat = matrix(0, number_lambdas, number_lambdag)
   rss_mat = matrix(0, number_lambdas, number_lambdag)
   iter_mat = matrix(0, number_lambdas, number_lambdag)
   for (lambda1_index in 1 : number_lambdas)
   {
      fslm[[lambda1_index]] = list()
      lambda1 = lambdas[lambda1_index]
      # generate initial values with lambdas fixed
      new_beta = rep(0, try_number_nodes)
      new_gamma = initial_gamma_slm(predictors[, -1], keep_residuals, try_number_nodes)
      beta = c(keep_beta, new_beta)
      gamma = cbind(keep_gamma, new_gamma)
      alpha = keep_alpha
      number_nodes = length(beta)
      active_nodes = 1 : number_nodes
      number_active_nodes = number_nodes
      planes = matrix(0, sample_size, number_active_nodes)
      activated_planes = matrix(0, sample_size, number_active_nodes)
      activated_planes_derivative = matrix(0, sample_size, number_active_nodes)
      fitted_values = rep(alpha, sample_size)
      for (m in FT(1, number_active_nodes))
      {
         planes[, m] = predictors %*% gamma[, m]
         activated_planes[, m] = bspline(planes[, m])
         activated_planes_derivative[, m] = bspline_derivative(planes[, m])
         fitted_values = fitted_values + beta[m] * activated_planes[, m]
      }
      residuals = response - fitted_values
      for (lambda2_index in 1 : number_lambdag)
      {
         lambda2 = lambdag[lambda2_index]
         store_rss = 0.5 * sum(residuals^2) + lambda1 * sum(abs(beta)) + lambda2 * sqrt(sum(gamma[-1, (beta != 0)]^2))
         for (iter in 1 : max_iterations)
         {
            # update alpha
            partial_residuals = residuals + alpha
            alpha = mean(partial_residuals)
            residuals = partial_residuals - alpha
            for (a in FT(1, number_active_nodes))
            {
               m = active_nodes[a]
               # update beta
               partial_residuals = residuals + beta[m] * activated_planes[, m]
               beta[m] = univariate_lasso_rss(partial_residuals, activated_planes[, m],
                                              lambda1, threshold)
               residuals = partial_residuals - beta[m] * activated_planes[, m]
               # update gamma 
               if (beta[m] != 0)
               {
                  for (j in 1 : number_predictors)
                  {
                     if (mean(gamma[j, ]^2) > 0)
                     {
                        partial_residuals = residuals + beta[m] * activated_planes[, m]
                        partial_plane = planes[, m] - gamma[j, m] * predictors[, j]
                        derivative_predictor = activated_planes_derivative[, m] * 
                           predictors[, j] * beta[m]
                        partial_gamma = residuals + derivative_predictor * gamma[j, m]
                        if (j == 1)
                           gamma[j, m] = univariate_lasso_rss(partial_gamma, derivative_predictor,
                                                              0, threshold)
                        else
                        {
                           w = sum(gamma[j, -m]^2)
                           gamma[j, m] = group_lasso_rss_smoothing(residuals, derivative_predictor, 
                                                                   w, gamma[j, m], lambda2, 
                                                                   step_size, 1e-10)
                        }
                        planes[, m] = partial_plane + gamma[j, m] * predictors[, j]
                        activated_planes[, m] = bspline(planes[, m])
                        activated_planes_derivative[, m] = bspline_derivative(planes[, m])
                        residuals = partial_residuals - beta[m] * activated_planes[, m]
                     }
                  }
               }
            }
            # pruning
            active_nodes = (1 : number_nodes)[beta != 0]
            number_active_nodes = length(active_nodes)
            # check variable selection
            if (sum(beta != 0) == 0)
               break
            for (j in 2 : number_predictors)
            {
               if (mean(gamma[j, (beta != 0)]^2) < 1e-02)
                  gamma[j, ] = rep(0, number_nodes)
            }
            fitted_values = rep(alpha, sample_size)
            for (a in FT(1, number_active_nodes))
            {
               m = active_nodes[a]
               planes[, m] = predictors %*% gamma[, m]
               activated_planes[, m] = bspline(planes[, m])
               activated_planes_derivative[, m] = bspline_derivative(planes[, m])
               fitted_values = fitted_values + beta[m] * activated_planes[, m]
            }
            residuals = response - fitted_values
            rss = 0.5 * sum(residuals^2) + lambda1 * sum(abs(beta)) + lambda2 * sqrt(sum(gamma[-1, (beta != 0)]^2))
            if (verbose)
               cat("\n", lambda1_index, "lambda1 th,", lambda2_index, "lambda2 th \n", 
                   iter, "th rss =", store_rss)
            # check convergence
            if (abs(rss - store_rss) < epsilon)
               break
            else
               store_rss = rss
         }
         # save results
         fitted_values = rep(alpha, sample_size)
         for (a in FT(1, number_active_nodes))
         {
            m = active_nodes[a]
            planes[, m] = predictors %*% gamma[, m]
            activated_planes[, m] = bspline(planes[, m])
            activated_planes_derivative[, m] = bspline_derivative(planes[, m])
            fitted_values = fitted_values + beta[m] * activated_planes[, m]
         }
         residuals = response - fitted_values
         rss = 0.5 * sum(residuals^2)
         # keep results for initial values at the first lambdag
         if (lambda2_index == 1)
         {
            keep_residuals = residuals
            keep_beta = beta[beta != 0]
            keep_alpha = alpha
            keep_gamma = gamma[, beta != 0]
         }
         TF_active = (beta != 0)
         active_predictors = rowSums(gamma[-1, ]) != 0
         number_active_predictors = sum(active_predictors)
         dimension = sum(gamma[, TF_active] != 0)
         fslm[[lambda1_index]][[lambda2_index]] = list(fitted_values = fitted_values, 
                                                       activated_planes = activated_planes,
                                                       alpha = alpha,
                                                       beta = beta[TF_active], 
                                                       gamma = matrix(gamma[, TF_active], number_predictors, sum(TF_active)),
                                                       number_active_nodes = sum(TF_active),
                                                       active_predictors = active_predictors,
                                                       number_active_predictors = number_active_predictors)
         number_active_node_mat[lambda1_index, lambda2_index] = sum(TF_active)
         bic_mat[lambda1_index, lambda2_index] = sample_size * log(rss) + 
            (number_active_predictors) * sum(TF_active) * log(sample_size)
         aic_mat[lambda1_index, lambda2_index] = sample_size * log(rss) + 
            (number_active_predictors) * sum(TF_active) * 2
         mix_mat[lambda1_index, lambda2_index] = sample_size * log(rss) + 
            (number_active_predictors) * (2 + log(sample_size)) / 2
         iter_mat[lambda1_index, lambda2_index] = iter
         rss_mat[lambda1_index, lambda2_index] = rss
      }
   }
   fslm$number_active_node_mat = number_active_node_mat
   fslm$bic_mat = bic_mat
   fslm$aic_mat = aic_mat
   fslm$mix_mat = mix_mat
   fslm$iter_mat = iter_mat
   fslm$rss_mat = rss_mat
   return(fslm)
}

# prediction function
prediction_slm = function(fit, points)
{
   n = nrow(grid_points)
   p = ncol(grid_points)
   act_planes = matrix(0, n, fit$number_active_nodes)
   fitted_values = rep(fit$alpha, n)
   for (m in 1 : fit$number_active_nodes)
   {
      plane = cbind(1, grid_points) %*% fit$gamma[, m]
      act_planes[, m] = bspline(plane)
      fitted_values = fitted_values + fit$beta[m] * act_planes[, m]
   }
   return(fitted_values)
}

# generate initial gammas using msir
initial_gamma_slm = function(x, y, number_nodes, directions = 1, s = 1)
{
   p = ncol(x)
   number_nodes_dir = number_nodes / length(directions)
   gamma = matrix(0, p + 1, 0)
   msir_fit = msir(x = x, y = y, nslices = 10, G = rep(10, 10))
   for (d in directions)
   {
      dir = msir_fit$std.basis[, d]
      # new predictors z = x' direction1 -> tau0 + tau1 z
      # find the appropriate tau0, tau1 in terms of z
      z = x %*% dir
      range_z = abs(max(z) - min(z))
      width = runif(number_nodes_dir, min(abs(diff(z))), range_z)
      centers = runif(number_nodes_dir, min(z), max(z))
      tau1 = 2 / width
      tau0 = - centers * tau1
      gamma0 = tau0
      gamma_plus = matrix(0, p, number_nodes_dir)
      for (k in 1 : p)
         gamma_plus[k, ] = tau1 * dir[k]
      gamma = cbind(gamma, rbind(gamma0, gamma_plus))
   }
   return(as.matrix(gamma))
}

# prediction on new predictors
predict_slm = function(fit, x)
{
   n = nrow(new_x)
   fitted_values = rep(fit$alpha, n)
   for (m in 1 : fit$number_nodes)
   {
      if (fit$beta[m] != 0)
      {
         plane = cbind(1, new_x) %*% fit$gamma[, m]
         activated_planes = bspline(plane)
         fitted_values = fitted_values + fit$beta[m] * activated_planes
      }
   }
   return(fitted_values)
}

# activate a plane using a linear bspline
bspline = function(h)
{
   a = rep(0, length(h))
   left = -1 <= h & h < 0
   a[left] = h[left] + 1
   right = 0 <= h & h < +1
   a[right] = 1 - h[right]
   return(a)
}

# derivative of a linear spline
bspline_derivative = function(h)
{
   da = rep(0, length(h))
   da[-1 <= h & h < 0] = +1
   da[0 <= h & h < +1] = -1
   return(da)
}

# penalization function

# univariate_lasso computes the minimizer zstar of
# q(z) = 0.5 * sum((y - z * x)^2) + lambda * abs(z)
# note zstar = soft_thresholding(b / a, lambda / a)
# for a = sum(x^2) and b = sum(x * y)
univariate_lasso_rss = function(y, x, lambda, threshold)
{
   a = sum(x^2)
   # if |a| is small, then q(z) = 0.5 * sum(y^2) + lambda * abs(z)
   # is minimized by zero, i.e. zstar = 0
   if (a < threshold)
      zstar = 0
   else
   {
      # if |a| is not so small, the we soft threshold.
      b = sum(x * y)
      zstar = soft_thresholding(b / a, lambda / a)
   }
   if (abs(zstar) < threshold)
      zstar = 0
   return(zstar)
}

# minimizes 0.5 * (z - b)^2 + lambda abs(z)   
soft_thresholding = function(b, lambda)
{
   if (b > lambda)
      return(b - lambda)
   else if (b < -lambda)
      return(b + lambda)
   else return(0)
}

# g(z) = 0.5 * sum((y - zx)^2) + lambda * sqrt(z^2 + w)
group_lasso_rss = function(y, x, w, z, lambda, threshold)
{
   if (sum(x^2) < 1e-20)
      return(0)
   prime1_g = -sum((y - z * x) * x) + (lambda * z) / sqrt(z^2 + w)
   prime2_g = sum(x^2) + (lambda * w) / (z^2 + w)^1.5
   updated_z = z - prime1_g / prime2_g
   return(updated_z)
}

# g(z) = 0.5 * sum((y - beta sigma(zx_i + k))^2) + lambda * sqrt(z^2 + w)
group_lasso_rss_smoothing = function(residuals, dev_predictors, w, z, lambda, step_size, threshold)
{
   # if (sum(dev_predictors^2) < 1e-20)
   #    return(0)
   if (sqrt(sum(w + z^2)) > 0.5)
   {
      prime1_g = -sum(residuals * dev_predictors) + (lambda * z) / sqrt(z^2 + w)
      prime2_g = sum(dev_predictors^2) + (lambda * w) / (z^2 + w)^1.5
   }
   else
   {
      prime1_g = -sum(residuals * dev_predictors) + (lambda * z)
      prime2_g = sum(dev_predictors^2) + lambda
   }
   updated_z = z - step_size * prime1_g / prime2_g
   return(updated_z)
}

# utility functions

# FT enumerates integers [from, to] only when from <= to.
FT = function(from, to)
{
   if (from > to)
      ft = NULL
   else
      ft = from : to
   return(ft)
}

test_stop = function(x)
{
   if (x < 0)
      stop(x < 0)
   if (x > 0)
      cat("x > 0", "\n")
}

#  copyright (c) sdmlab
#  korea university, seoul, republic of korea.
#  email for maintenance: jykoo@korea.ac.kr
#  work note (by jyk)
#  version 1.0
#     programing is intiated
#  version 1.1
#     'initial_gamma_slm' is added to generate initial values for gamma
#  version 1.2
#     The l-1 penalization method for beta is used 
#     along a set of complexity parameters for node selection
#  version 1.3
#     proceed algorithm from the largest complexity to the smallest
#     the initial values are updated applying msir method based on residuals
#     when nodes are added
#  version 1.3.3
#     M random initial values added when lambda changes