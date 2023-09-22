import numpy as np

# One-out-of-K ransformation
def one_out_of_k_encoding(column_vector):
    num_samples = len(column_vector)
    num_classes = len(np.unique(column_vector))
    one_hot_encoding = np.zeros((num_samples,num_classes))
    one_hot_encoding[np.arange(num_samples), column_vector] = 1
    
    return one_hot_encoding