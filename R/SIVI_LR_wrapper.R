#####################
tf$reset_default_graph()


Exponential = tf$contrib$distributions$Exponential(rate=1.0)
Normal = tf$contrib$distributions$Normal(loc=0., scale=1.)
Mvn = tf$contrib$distributions$MultivariateNormalDiag
Bernoulli = tf$contrib$distributions$Bernoulli


samplelr_n = function(mu, sigma) {
  eps = tf$random_normal(shape=tf$shape(mu))
  z = mu+tf$matmul(eps, sigma)
  return(z)
}


samplelr_hyper <- function(noise_dim, K, z_dim, reuse = F) {
  with(tf$variable_scope("hyper_q") %as% scope, {
    if (reuse) {scope$reuse_variables()}
    e2 = tf$random_normal(shape = c(K,noise_dim))
    h2 = tf$contrib$slim$stack(e2,tf$contrib$slim$fully_connected, as.integer(c(100,200,100)))
    mu = tf$reshape(tf$contrib$slim$fully_connected(h2, z_dim, activation_fn = NULL, scope='implicit_hyper_mu'), as.integer(c(-1,z_dim)))
  })
  return(mu)
}
#######################


#' SIVI for Logistic Regression
#'
#' @description This function is used for the implementation of the SIVI's algorithm (semi-implicit variational inference) in the
#' logistic regression set-up. It adopts the adam optimizer for the covariance matrix and uses the gradient descent optimizer for
#' the Neural Network in the implicit part. The iter number can be tuned manually, and the output is the sampling from the posterior
#' of the latent variables. In this configuration, we have the number of latent variables equal to the numbers of parameters input
#' by X.
#'
#' @param X the matrix for training
#' @param Y the corresponding labels
#' @param noise_dim the input noise dimension
#' @param K the added terms
#' @param J the sampling numbers of latent variables
#' @param n_iter the whole iteration number
#' @param inner_iter the iteration number for the nural network which reside in the inner part of the iteration and is supposed to be smaller than the iter_num.
#' @param pos_num the sampling numbers of the posterior
#' @param merge when merge equals one means using the lower bound, when it equals zero means using the upper bound
#' @param lr the learning rate inside the 200 times iteration
#' @param lr2 the learning rate outside the 200 times iteration
#'
#' @return the sampling of posterior for the latent variables
#' @export
#'
#' @examples sivi_lr(waveform$X.train, waveform$y.train)
sivi_lr <- function(X, Y, noise_dim = 20, K = 50, J = 100, merge = 1, lr = 0.0001, lr2 = 0.01,
                    n_iter = 500, inner_iter = 200, pos_num = 1000) {
  # Compatibility check
  if(dim(X)[1] != length(Y)){
    stop("the dimension of x and y are not compitible")
  }

  if(inner_iter > n_iter){stop("the inner_iter should not exceed the n_iter.")}

  if(any(Y > 1 |  Y < 0)){
    warning("the value of y should be between 0 and 1. This may affect the result.")
    Y[Y > 1] = 1
    Y[Y < 0] = 0
  }

  ####################

  X_train = X
  y_train = as.integer(Y)
  noise_dim = as.integer(noise_dim)
  K = as.integer(K)
  J = as.integer(J)
  merge = as.integer(merge)
  lr = tf$constant(lr)
  lr2 = tf$constant(lr2)
  alpha = 0.01

  ####################

  N = as.integer(dim(X_train)[1])
  P = as.integer(dim(X_train)[2])


  fff = tf$get_variable("z", dtype = tf$float32,
                        initializer = tf$zeros(as.integer((P+1)*P/2)) + 0.2)
  chol_cov = tf$contrib$distributions$fill_triangular(fff)
  covariance = tf$matmul(chol_cov, tf$transpose(chol_cov))

  inv_cov = tf$matrix_inverse(covariance)
  inv_cov_1 = tf$expand_dims(inv_cov, axis = as.integer(0))
  inv_cov_2 = tf$tile(inv_cov_1, list(as.integer(J+1), as.integer(1), as.integer(1)))

  log_cov_det = tf$log(tf$matrix_determinant(covariance))

  scale = tf$placeholder(tf$float32)

  x = tf$placeholder(tf$float32, list(N, P), name='data_x')
  y = tf$placeholder(tf$float32, list(N), name='data_y')
  psi_sample = tf$squeeze(samplelr_hyper(noise_dim, K, P))
  z_sample = samplelr_n(psi_sample,tf$transpose(chol_cov))

  psi_star_0 = samplelr_hyper(noise_dim, J, P, reuse=T)
  psi_star_1 = tf$expand_dims(psi_star_0, axis = as.integer(1))
  psi_star_2 = tf$tile(psi_star_1, list(as.integer(1), K, as.integer(1)))

  psi_star = tf$cond(tf$constant(merge>0, dtype = tf$bool), function() tf$concat(list(psi_star_2, tf$expand_dims(psi_sample, axis = as.integer(0))), as.integer(0)), function() psi_star_2)

  z_sample_0 = tf$expand_dims(z_sample, axis = as.integer(0))
  z_sample_1 = tf$cond(tf$constant(merge>0, dtype = tf$bool), function() tf$tile(z_sample_0,list(as.integer(J+1), as.integer(1), as.integer(1))),function() tf$tile(z_sample_0,list(as.integer(J), as.integer(1), as.integer(1))))


  xvx = tf$matmul(z_sample_1 - psi_star, inv_cov_2)*(z_sample_1 - psi_star)
  ker = tf$transpose(-0.5*tf$reduce_sum(xvx, as.integer(2)))
  log_H = tf$reduce_logsumexp(ker, axis = as.integer(1), keep_dims = T) - tf$log(tf$cast(J, tf$float32) + 1.0) - 0.5*log_cov_det

  log_P = scale*tf$reduce_sum(-tf$nn$softplus(-tf$matmul(z_sample, tf$transpose(x))*y), axis = as.integer(1), keep_dims = T) + (-0.5)*alpha*tf$reduce_sum(tf$square(z_sample), axis = as.integer(1), keep_dims = T)

  loss = tf$reduce_mean(log_H - log_P)

  nn_var = tf$get_collection(tf$GraphKeys$GLOBAL_VARIABLES, scope='hyper_q')
  train_op1 = tf$train$AdamOptimizer(learning_rate = lr)$minimize(loss, var_list = nn_var)

  train_op2 = tf$train$GradientDescentOptimizer(learning_rate = lr2)$minimize(loss, var_list=list(fff))

  init_op = tf$global_variables_initializer()


  #########################

  J = tf$constant(J)
  merge = tf$constant(merge)

  sess = tf$InteractiveSession()
  sess$run(init_op)

  record = numeric()

  for (i in 1:n_iter) {
    result = sess$run(list(train_op1, loss), dict(x = X_train, y = y_train, lr = 0.01*(0.9**(i/100)), J = as.integer(100), merge = as.integer(1), scale = 1.0))

    if(i < inner_iter) {
      result = sess$run(list(train_op2, loss), dict(x = X_train, y = y_train, lr2 = 0.001*(0.9**(i/100)), J = as.integer(100), merge = as.integer(1), scale = 1.0))
    }

    record = append(record, result[[2]])
    if(i %% round(n_iter/10) == 0) {
      cat("iter:", i+1, "loss =", mean(record, na.rm = T), ', std =', pracma::std(record), "\n")
      record = numeric()
    }
  }

  ##########################

  # sample from the learned posterior
  theta_hive = matrix(0, pos_num, P)
  for (i in 1:1000) {
    r = sess$run(z_sample)
    theta_hive[i,] = r[1,]
  }

  # returning the sampled posterior
  return(list("sample_pos" = theta_hive))

}
