######################
# including some packages
library(tensorflow)
library(pracma)


######################

slim = tf$contrib$slim
Exponential = tf$contrib$distributions$Exponential(rate=1.0)
Normal = tf$contrib$distributions$Normal(loc=0., scale=1.)
Mvn = tf$contrib$distributions$MultivariateNormalDiag
Bernoulli = tf$contrib$distributions$Bernoulli


tf$reset_default_graph()


samplelr_n = function(mu, sigma) {
  eps = tf$random_normal(shape=tf$shape(mu))
  z = mu+tf$matmul(eps, sigma)
  return(z)
}


samplelr_hyper <- function(noise_dim, K, z_dim, reuse=False) {
  with(tf$variable_scope("hyper_q") %as% scope, {
    if (reuse) {scope$reuse_variables()}
    e2 = tf$random_normal(shape = c(K,noise_dim))
    h2 = slim$stack(e2,slim$fully_connected, as.integer(c(100,200,100)))
    mu = tf$reshape(slim$fully_connected(h2, z_dim, activation_fn = None, scope='implicit_hyper_mu'), as.integer(c(-1,z_dim)))
    })
  return(mu)
}


###################################
# Loading the data for testing the model into X_train, x_test, y_train, y_test.

path = "data path"

matdata = read.csv(path)

X_train = matdata['X_train']
X_test = matdata['X_test']
y_train = as.integer(drop(matdata['y_train']))
y_test = as.integer(drop(matdata['y_test']))

#transforming the labels of data for the logistic regression
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1

###################################
#Stored Mean-field and MCMC results
theta_mf = t(matdata['Beta_VB_sample'])
theta_mcmc = t(matdata['BetaMCMC'])

###################################

# Setting and getting some basic parameters
N = as.integer(dim(X_train)[1])
P = as.integer(dim(X_train)[2])
noise_dim = as.integer(20)
J = tf$placeholder(tf$int32)


fff = tf$get_variable("z", dtype = tf$float32,
                      initializer = tf$zeros(((P+1)*P/2)+0.2))
chol_cov = fill_triangular(fff)
covariance = tf$matmul(chol_cov, tf$transpose(chol_cov))

inv_cov = tf$matrix_inverse(covariance)
inv_cov_1 = tf$expand_dims(inv_cov, axis=0)
inv_cov_2 = tf$tile(inv_cov_1, list(J+1, as.integer(1), as.integer(1)))

log_cov_det = tf$log(tf$matrix_determinant(covariance))


K = as.integer(50)
scale = tf$placeholder(tf$float32)

x = tf$placeholder(tf$float32, list(N, P), name='data_x')
y = tf$placeholder(tf$float32, list(N), name='data_y')
psi_sample = tf$squeeze(sample_hyper(noise_dim, K, P))
z_sample = sample_n(psi_sample,tf$transpose(chol_cov))

psi_star_0 = sample_hyper(noise_dim, J, P, reuse=True)
psi_star_1 = tf$expand_dims(psi_star_0, axis=1)
psi_star_2 = tf$tile(psi_star_1, list(1, K, 1))


merge = tf$placeholder(tf$int32)
psi_star = tf$cond(merge>0, function() tf$concat(list(psi_star_2, tf$expand_dims(psi_sample, axis=0)), 0), function() psi_star_2)

z_sample_0 = tf$expand_dims(z_sample, axis=0)
z_sample_1 = tf$cond(merge>0, function() tf$tile(z_sample_0,list(J+1, as.integer(1), as.integer(1))),function() tf$tile(z_sample_0,list(J, as.integer(1), as.integer(1))))


xvx = tf$matmul(z_sample_1 - psi_star, inv_cov_2)*(z_sample_1 - psi_star)
ker = tf$transpose(-0.5*tf$reduce_sum(xvx, as.integer(2)))
log_H = tf$reduce_logsumexp(ker, axis=1, keep_dims = True) - tf$log(tf$cast(J, tf$float32)+1.0) - 0.5*log_cov_det

log_P = scale*tf$reduce_sum(-tf$nn$softplus(-tf$matmul(z_sample, tf$transpose(x))*y), axis=1, keep_dims = True) + (-0.5)*alpha*tf$reduce_sum(tf$square(z_sample), axis = 1, keep_dims = True)


loss = tf$reduce_mean(log_H - log_P)

nn_var = tf$get_collection(tf$GraphKeys$GLOBAL_VARIABLES, scope='hyper_q')
lr = tf$constant(0.0001)
train_op1 = tf$train$AdamOptimizer(learning_rate = lr)$minimize(loss, var_list = nn_var)

lr2=tf$constant(0.01)
train_op2 = tf$train$GradientDescentOptimizer(learning_rate = lr2)$minimize(loss, var_list=list(fff))

init_op = tf$global_variables_initializer()

###################################

sess = tf$InteractiveSession()
sess$run(init_op)

record = numeric()

for (i in 1:5000) {
  result = sess$run(list(train_op1, loss), dict(x = X_train, y = y_train, lr = 0.01*(0.9**(i/100)), J = as.integer(100), merge = as.interger(1), scale = 1.0))

  if(i < 2000) {
    result = sess$run(list(train_op2, loss), dict(x = X_train, y = y_train, lr2 = 0.001*(0.9**(i/100)), J = as.integer(100), merge = as.integer(1), scale = 1.0))

    record = append(record, result[[2]])
  }
  if(i %% 100) {
    cat("iter:", i+1, "cost=", mean(record), ',', std(record))
    record = numeric()
  }
}

####################################

# sample from the learned posterior
theta_hive = matrix(0, 1000, P)
for (i in 1:1000) {
  r = sess$run(z_sample)
  theta_hive[i,] = r[0,]
}


####################################

evaluate = function(theta, X_test, y_test) {
  M = dim(theta)[1]
  n_test = length(test)
  prob = matrix(0, n_test, M)
  blr = matrix(0, n_test, M)
  for (t in 1:M) {
    coff = rowSums(-1*(repmat(theta[t, ], n_test, 1) %*% X_test))
    blr[, t] = rep(1, n_test)/(1 + exp(coff))
    coff1 = y_test %*% rowSums(-1*(repmat(theta[t, ], n_test, 1)) %*% X_test)
    prob[, t] = rep(1, n_test) / (1 + exp(coff1))
  }
  prob = rowMeans(prob)
  scatter.smooth(rowMeans(blr), rowMeans(blr))
  return(list("mean" = rowMeans(blr), "std" = apply(blr, 1, std)))
}

evaluate(theta_mcmc, X_test, y_test)
evaluate(theta_hive, X_test, y_test)


####################################

