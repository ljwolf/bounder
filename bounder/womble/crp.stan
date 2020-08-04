data{
  int<lower=0> C;//num of cluster
  int<lower=0> N;//n samples
  int<lower=0> P;//n features
  row_vector[P] X[N];
}

parameters {
  ordered[C] mu_cl[P]; //cluster mean
  real <lower=0,upper=1> v[C];
  vector<lower=0>[C] sigma_cl[P]; // error scale
  //real<lower=0> alpha; // hyper prior DP(alpha,base)
}

transformed parameters{
  vector[C] pi;
  pi[1] = v[1];
  // stick-break process based on The BUGS book Chapter 11 (p.294)
  for(j in 2:(C-1)){
      pi[j]= v[j]*(1-v[j-1])*pi[j-1]/v[j-1]; 
  }
  pi[C]=1-sum(pi[1:(C-1)]); // to make a simplex.
}

model {
  real alpha = 1;
  real ps[N,C];
  real assignments[N];
  
  for (k in 1:C){
    sigma_cl[,k] ~ cauchy(0,5);
    mu_cl[,k] ~ normal(0,1);
  }
  v ~ beta(1,alpha);
  
  for(i in 1:N){
    for(c in 1:C){
      ps[i,c]=log(pi[c])+normal_lpdf(X[i] | mu_cl[,c], sigma_cl[,c]);
    }
    target += log_sum_exp(ps[i]);
  }
}

generated quantities{
    vector[N] assignments;
    
    for (i in 1:N){
        real cluster_min = 100000;
        real current;
        for (c in 1:C){
            current = dot_self(to_vector(mu_cl[,c]) - to_vector(X[i]));
            if(current < cluster_min){
                assignments[i] = c;
                cluster_min = current;
            }
        }
    }
}