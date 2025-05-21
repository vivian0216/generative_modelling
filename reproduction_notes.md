# Paper replication issues
- Algorithm 1 in the paper uses the incorrect sign for Lgen.
- The paper does not mention batched gradient descent, even though it is used in their code. Batch size is not listed as hyperparameter.
- The paper does not mention initializing the replay buffer fully with random samples, and updating them inplace when performing sgld and adding them back to the buffer. This is done in their code.