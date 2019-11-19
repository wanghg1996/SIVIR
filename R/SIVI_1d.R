slim = tf$contrib$slim
Exponential = tf$contrib$distributions$Exponential(rate = 1.0)
Normal1 = tf$contrib$distributions$Normal(loc = -2.0, scale = 1.0)
Normal2 = tf$contrib$distributions$Normal(loc = 2.0, scale = 1.0)
Normal = tf$contrib$distributions$Normal(loc = 0.0, scale = 1.0)


sample_n <- function(mu, sigma) {
  eps = tf$random_normal(shape = tf$shape(mu))
  z = mu + eps*sigma
  return(z)
}


sample_hyper <- function(noise_dim, K, reuse = F) {
  z_dim = 1
  with(tf$variable_scope("hyper_q", reuse = tf$AUTO_REUSE) %as% scope, {
    if(reuse){scope$reuse_variables()}
    e2 = tf$random_normal(shape = as.integer(c(K, noise_dim)))
    input_ = e2
    h2 = slim$stack(input_, slim$fully_connected, as.integer(c(20, 40, 20)))
    mu = tf$reshape(slim$fully_connected(h2, as.integer(z_dim), activation_fn = NULL, scope = "implicit_hyper_mu"), as.integer(c(-1.0, 1.0)))
  })
  return(mu)
}


#################################

data_p = dict("1" = "guassian", "2" = "laplace", "3" = "gmm")
data_number = 3
target = darget = data_p[data_number]

#################################

noise_dim = 10
K = 20

psi_sample = sample_hyper(noise_dim, K)

sigma = tf$constant(0.2)
z_sample = sample_n(psi_sample, sigma)


J = tf$placeholder(tf$int32, shape = NULL)
psi_star = tf$transpose(sample_hyper(noise_dim, J, reuse = T))

merge = tf$placeholder(tf$int32, shape = NULL)
psi_star = tf$cond(merge>0, function() tf$concat(list(psi_star, tf$transpose(psi_sample)), as.integer(1)), function() psi_star)


log_H = tf$log(tf$reduce_mean(tf$exp(-0.5*tf$square(z_sample - psi_star)/tf$square(sigma)), axis = as.integer(1),
                              keep_dims = T))

if (target == "guassian") {
  log_p = -tf$log(3.0) - 0.5*tf$square(z_sample)/tf$square(3.0) #guassian
} else if (target == "laplace") {
  log_P = -0.5*tf$abs(z_sample) #laplace(mu=0,b=2)
} else if (target == "gmm") {
  log_P = tf$log(0.3*tf$exp(-tf$square(z_sample + 2)/2) + 0.7*tf$exp(-tf$square(z_sample - 2)/2))
} else {
  stop("No pre-defined target distribution, you can write your own log(PDF). ")
}


loss = tf$reduce_mean(log_H - log_P)

nn_var = slim$get_model_variables()
lr = tf$constant(0.01)
train_op1 = tf$train$AdamOptimizer(learning_rate = lr)$minimize(loss, var_list = nn_var)

init_op = tf$global_variables_initializer()


############################
