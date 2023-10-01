import numpy as np

#input is a matrix
def basic_info(matrix):
    
    matrix = np.array(matrix)


    for col_index in [0,1,3,5,6,7,8,9,10,11,12]:
        current_column = matrix[:, col_index]
        unique_values = np.unique(current_column)
        unique_values.sort()
        value_to_rank = {value: rank +1 for rank, value in enumerate(unique_values)}
        matrix[:, col_index] = np.vectorize(value_to_rank.get)(current_column)

    

    mean = np.mean(matrix, axis=0)
    median = np.median(matrix, axis=0)
    std_dev = np.std(matrix, axis=0)
    min_val = np.min(matrix, axis=0)
    max_val = np.max(matrix, axis=0)
    matrix_info = np.vstack(( mean, median, std_dev, min_val, max_val))
    return matrix_info
    


