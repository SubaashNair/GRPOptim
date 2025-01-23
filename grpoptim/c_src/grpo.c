#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// structure to hold batch data
typedef struct{
    double* log_probs_old;
    double* log_probs_ref;
    double* rewards;
    int group_size;
} GRPOBatch;

// compute advantages (standardized rewards)
void compute_advantages(double* rewards, double* advantages, int group_size){
    double mean = 0.0, std = 0.0;

    // calculate mean
    for (int i=0; i<group_size; i++) mean += rewards[i];
    mean/= group_size;

    // calculate standard deviation
    for (int i = 0; i<group_size; i++) std += pow(rewards[i]-mean,2);
    std = sqrt(std/group_size + 1e-8);

    // compute advantages
    for (int i = 0; i < group_size; i++){
        advantages[i] = (rewards[i] - mean) /std;
    }
}

// computing GRPO loss and gradients
void grpo_loss(
    GRPOBatch* batch,
    double* log_probs_new,
    double* loss,
    double* grad,
    double epsilon,
    double beta
){
    int G = batch->group_size;
    double* advantages = (double*)malloc(G* sizeof(double));
    compute_advantages(batch->rewards, advantages,G);

    double total_loss = 0.0;

    for (int i=0; i<G; i++){
        double ratio = exp(log_probs_new[i] - batch->log_probs_old[i]);
        double clipped_ratio = fmin(fmax(ratio,1-epsilon),1+epsilon);
        double surr = fmin(ratio * advantages[i], clipped_ratio*advantages[i]);

        // KL divergence penalty simplified
        double kl_term = exp(batch->log_probs_ref[i] - log_probs_new[i]) - (batch->log_probs_ref[i] - log_probs_new[i]) -1;
        total_loss += surr - beta * kl_term;
        grad[i] = (surr - beta*kl_term)/G;
    }
    *loss =- total_loss /G;
    free(advantages);
}