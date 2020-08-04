functions{
    int has_neighbor_in_cluster(int i, int c, matrix connectivity, vector assignments){
        int n_samples = length(assignments);
        for (j in 1:n_samples){
            if (connectivity[i,j] > 0){
                if (assignments[j] == c){
                    return 1;
                }
            }
        }
        return 0;
    }

    int can_assign(int i, int c, matrix connectivity, vector assignments){
        int cluster_is_populated = (sum(assignments == c) > 0);
        if (cluster_is_populated){
            return has_neighbor_in_cluster(i, c, connectivity, assignments);
        }
        else{
            return 1;
        }
    }
}

data {
    int<lower=0> n_clusters;
    int<lower=0> n_samples;
    int<lower=0> n_features;
    
    matrix[n_samples, n_features] X;
    matrix[n_samples, n_samples] connectivity;
    
}
parameters {
    ordered[n_features] cluster_mean[n_clusters];
    real<lower=0, upper=1> potentials[n_clusters];
    real<lower=0> cluster_variance[n_clusters];
}

transformed parameters{
    simplex [n_clusters] potentials_normed;
    
    potentials_normed[1] = potentials[1];
    for (j in 1:(n_clusters - 1)){ // stickbreaking, BUGS book, Ch. 11 p 294
        potentials_normed[j] = (potentials[j] * (1 - potentials[j-1])
                                * potentials_normed[j-1] / potentials[j-1]);
    potentials_normed[n_clusters] = 1 - sum(potentials_normed[1:(n_clusters - 1)]);
    }
}

model {
    real alpha = 1;
    real cluster_probability_for_i[n_clusters];
    vector[n_samples] assignments
    cluster_variance ~ cauchy(2);
    cluster_mean ~ normal(0,1);
    potentials ~ beta(1, alpha);
    
    for (i in 1:n_samples){
        for (c in 1:n_clusters){
            if(i==1){
                can_assign = 1
            }
            else{
                can_assign = has_neighbor_in_cluster
            }
            cluster_probability_for_i[c] = log(potentials_normed[c])
                                           + (normal_lpdf(X[i] | cluster_mean[c], 
                                                           cluster_variance[c])
                                              * has_neighbor_in_cluster(i, c, 
                                                                        connectivity,
                                                                        assignment)); 
                                         
        }
        target += log_sum_exp(cluster_probability_for_i);
        assignment[i] = sort_indices_desc(cluster_probability_for_i)[1];
    }
}