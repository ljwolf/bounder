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
  int<lower=0> n_samples;
  int<lower=0> n_features;
  int<lower=0> n_clusters;
  matrix[n_samples, n_features] X;
  matrix[n_samples, n_samples] connectivity;
}

parameters{
  ordered[n_features] mu[n_clusters];
  row_vector<lower=0>[n_features] sigma[n_clusters];
  simplex[n_clusters] theta;
  simplex[n_clusters] p_assignments[n_samples];
}

transformed parameters{
    vector[n_clusters] log_theta = log(theta);
    vector[n_samples] assignments;
    for(i in 1:n_samples){
        assignments[i] = sort_indices_desc(p_assignments[i])[1];
        }
}

model {
  vector[n_clusters] log_p_assignments[n_samples] = log(p_assignments);
  for (c in 1:n_clusters){
      sigma[c] ~ cauchy(0,2);
      mu[c] ~ normal(0,1);
      }
  
  for (i in 1:n_samples){
      vector[n_clusters] raw_log_prob_i = log_theta;
      for (c in 1:n_clusters){
          raw_log_prob_i[c] += normal_lpdf(X[i] | mu[c], sigma[c]);
      }
      // this is what would need to be adjusted to enforce a spatial constraint.
      // log(p*constraint) = log(p) + log(constraint)
      log_p_assignments[i] = raw_log_prob_i + log(can_assign(i, c, connectivity, assignments));
      target += log_sum_exp(log_p_assignments[i]);
  }
}