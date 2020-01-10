def expend_vector_of_size(positional_vector, n):
    result = []
    for i in range(n):
        result.append(int(positional_vector.get(str(i), 0)))
    return result
