import numpy as np

#input is a matrix
def basic_info(matrix):
    
    matrix = np.array(matrix)
    
    mean = np.mean(matrix, axis=0)
    median = np.median(matrix, axis=0)
    std_dev = np.std(matrix, axis=0)
    min_val = np.min(matrix, axis=0)
    max_val = np.max(matrix, axis=0)
    matrix_info = np.vstack(( mean, median, std_dev, min_val, max_val))
    return matrix_info
    


