import numpy


def reverse_matrix(matrix):
    result = numpy.dot(1 / numpy.linalg.det(matrix), matrix.T)
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = int(result[i][j])
    return result


print(reverse_matrix(numpy.array([[1, 2], [2, 1]])))
