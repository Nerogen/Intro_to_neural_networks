import copy
import pickle

import numpy
from PIL import Image

from utils.config import first_size_side_of_cub, second_size_side_of_cub, error_of_learning, max_value_of_error, \
    incorrect_user_input_in_user_input_method, input_image_string, subtraction_error, incorrect_image


def transposition(matrix):
    """Transposition of some matrix
     :return matrix^T"""
    strings, columns = len(matrix), len(matrix[0])
    result = [[0] * strings for _ in range(columns)]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            result[j][i] = matrix[i][j]

    return result


def subtraction_of_matrix(first_matrix, second_matrix):
    """first_matrix - second_matrix"""
    if len(first_matrix) == len(second_matrix) and len(first_matrix[0]) == len(second_matrix[0]):
        result = copy.deepcopy(first_matrix)
        for i in range(len(first_matrix)):
            for j in range(len(first_matrix[i])):
                result[i][j] = first_matrix[i][j] - second_matrix[i][j]
        return result
    else:
        print(subtraction_error)


def print_array(matrix) -> None:
    """Print array in understandable format
    :return None"""
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print(matrix[i][j], end=' ')
        print()


def recovery_matrix(matrix):
    result = 1 / numpy.linalg.det(matrix) * matrix.T
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = int(result[i][j])
    return result


def multiplication_number_and_matrix(number: int, matrix):
    """:param number, matrix
    :return number*matrix"""
    result = copy.deepcopy(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            result[i][j] = matrix[i][j] * number

    return result


def multiplication(matrix1, matrix2):
    """get multiplication of two matrix
    :return matrix"""
    return list(
        list(sum([matrix1[i][k] * matrix2[k][j] for k in range(len(matrix2))]) for j in range(len(matrix2[0]))) for i
        in range(len(matrix1)))


def process(matrix, width: int, height: int):
    """Change all pixels with condition C[i][j][k] = ((2*C[i][j][k])/255) - 1
    :return matrix"""
    result = [
        [(((2 * matrix[i][j][0]) / 255) - 1), (((2 * matrix[i][j][1]) / 255) - 1), (((2 * matrix[i][j][2]) / 255) - 1)]
        for i in range(len(matrix)) for j in range(len(matrix))
    ]
    result = [result[i * width:(i + 1) * width] for i in range(height)]
    return result


def process_list_for_pillow_get_pixels(matrix):
    """Process data for pillow format"""
    result = []
    for i in matrix:
        for j in i:
            result.append(j[0])
            result.append(j[1])
            result.append(j[2])
    return result


def process_list_for_pillow(matrix, width: int, height: int, r: int, m: int, recovery):
    """Process data for pillow format(extend)"""
    result = copy.deepcopy(recovery)
    count_width = 0
    count_height = 0
    matrix = process_list_for_pillow_get_pixels(matrix)
    count_of_pos = 0

    while m + count_width <= width and r + count_height <= height:
        for i in range(count_width, m + count_width):
            for j in range(count_height, r + count_height):
                for pix in range(3):
                    result[i][j][pix] = matrix[count_of_pos]
                    count_of_pos += 1
        if m + count_width >= width and r + count_height < height:
            count_width = 0
            count_height += r
        else:
            count_width += r

    return result


def get_cubes(matrix, width: int, height: int, r: int, m: int):
    """cut images on cubes
    :return matrix_of_cubes"""
    result = []
    count_width = 0
    count_height = 0

    while m + count_width <= width and r + count_height <= height:
        for i in range(count_width, m + count_width):
            line = []
            for j in range(count_height, r + count_height):
                line.append(matrix[i][j])
            result.append(line[:])
        if m + count_width >= width and r + count_height < height:
            count_width = 0
            count_height += r
        else:
            count_width += r

    return result


def get_vector_from_cubes(matrix, r: int):
    """Make vector from cubes
    :return vectors from matrix (r, g, b)"""
    result = []
    count = 0
    vector = []
    for i in range(len(matrix)):
        count += 1
        for j in range(len(matrix[i])):
            vector.extend(matrix[i][j])
        if count == r:
            count = 0
            result.append(vector[:])
            vector = []

    return result


def recovery_color(matrix):
    """Change all pixels with condition C[i][j][k] = Cmax *(X`[i][j] + 1) /2
    :return matrix"""
    result = []
    rgb = []
    for i in range(len(matrix)):
        cube = []
        for j in range(len(matrix[i])):
            rgb.append(int((255 * (matrix[i][j] + 1)) / 2))
            if len(rgb) == 3:
                cube.append(rgb[:])
                rgb = []
        result.append(cube[:])

    return result


def converter_to_format_pillow_from_matrix(matrix):
    """present matrix in format pillow
    :return matrix"""
    result = [tuple(element) for pix in matrix for element in pix]
    return result


def make_image(image_mode, image_size, pixels, name) -> None:
    """Make image from getting data"""
    out_image = Image.new(image_mode, image_size)
    out_image.putdata(pixels)
    out_image.save(f'{name}.png')


def find_alpha(matrix) -> float:
    """Calculate alpha with condition 1/Sum(Y(i))^2 | 0 < a <= 0.01
    :return a"""
    result = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            result += matrix[i][j] ** 2

    return 1 / result


def find_first_weight(first_weight, a, Xi, deltaX, second_weight):
    """Calculate weight with condition W(t+1)=W(t)–a*[X(i)]^T *∆X(i)*[W`(t)]^T
    :return matrix"""
    return subtraction_of_matrix(first_weight, multiplication_number_and_matrix(
        a, multiplication(multiplication(transposition(Xi), deltaX), transposition(second_weight))))


def find_second_weight(second_weight, a, Yi, deltaX):
    """Calculate weight with conditionW`(t+1)=W`(t) – a`*[Y(i)]^T *∆X(i)
    :return matrix"""
    matrix = multiplication(transposition(Yi), deltaX)
    return subtraction_of_matrix(second_weight, multiplication_number_and_matrix(a, matrix))


def find_Eq(matrix) -> int:
    """Calculate E with condition E(q) = Sum (deltX(i) * deltX(i))
    :return number"""
    E = []
    for i in range(len(matrix)):
        result = 0
        for j in range(len(matrix[i])):
            result += matrix[i][j] ** 2
        E.append(result)

    return int(sum(E))


def find_first_weight_with_numpy(first_weight, a, Xi, deltaX, second_weight):
    """Calculate weight with condition W(t+1)=W(t)–a*[X(i)]^T *∆X(i)*[W`(t)]^T
    :return matrix"""
    return numpy.subtract(first_weight, a * numpy.dot(numpy.dot(Xi.T, deltaX), second_weight.T))


def find_second_weight_with_numpy(second_weight, a, Yi, deltaX):
    """Calculate weight with conditionW`(t+1)=W`(t) – a`*[Y(i)]^T *∆X(i)
    :return matrix"""
    matrix = numpy.dot(Yi.T, deltaX)
    return numpy.subtract(second_weight, a * matrix)


def pickle_dump_to_file(file_name, obj) -> None:
    """Serialization to file"""
    with open(f'{file_name}', 'wb') as file:
        pickle.dump(obj, file)


def get_weights(first_name: str, second_name: str) -> tuple:
    """Upload weights from files
    :return tuple(first_weights, second_weights)"""
    with open(f'{first_name}', 'rb+') as file:
        first_weights = pickle.load(file)

    with open(f'{second_name}', 'rb+') as file:
        second_weights = pickle.load(file)

    return first_weights, second_weights


def load_file(file_name: str):
    """Load pickle object
    :return obj"""
    with open(f'{file_name}', 'rb+') as file:
        obj = pickle.load(file)

    return obj


def open_and_process_image() -> tuple:  ##################
    """Open image and get pixels
    :return tuple(image_name, pixels, image, width, height)"""
    flag = True
    while flag:
        image_name = input(input_image_string)
        image = Image.open(f'{image_name}')
        pixels = list(image.getdata())
        width, height = image.size
        if width == height and width % 2 == 0:
            flag = False
        else:
            print(incorrect_image)

    return image_name, pixels, image, width, height


def user_input(height: int) -> tuple:
    """Get data from console and check of correction:
    :return tuple(n, m, p, e)"""
    flag = True
    while flag:
        n = int(input(first_size_side_of_cub))
        m = int(input(second_size_side_of_cub))
        p = n * m * 3
        e = int(input(error_of_learning))
        if n == m and n <= height and e < max_value_of_error and n % 2 == 0:
            flag = False
        else:
            print(incorrect_user_input_in_user_input_method)
    return n, m, p, e


def make_cubes_and_vectors(pixels, width: int, height: int, n: int, m: int) -> tuple:
    """Make pixel array and cut the on cubs, then making vectors
    :return tuple(vectors, copy_pix)"""
    # has 124 lists with 124 tuples with RGB
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    # C(k)[i][j] = ((2*C(k)[i][j])/256) - 1
    pixels = process(pixels, width, height)
    # get copy pixels for recovery image
    copy_pix = copy.deepcopy(pixels)
    # get cubes n x m
    cubes = get_cubes(pixels, width, height, n, m)
    # get vectors X(i)
    vectors = get_vector_from_cubes(cubes, n)

    return vectors, copy_pix


def learning(vectors, first_weight, second_weight) -> tuple:
    """Preparation step of learning, when we generated weights and get values from neural layer.
     :return tuple(delta_X, Yi, X_i, delta_X, first_weight, second_weight)"""
    # get Y(i)
    Yi = multiplication(vectors, first_weight)
    # get X`(i)
    X_i = multiplication(Yi, second_weight)
    # get ∆X(i) =  X`(i) - X(i)
    delta_X = subtraction_of_matrix(X_i, vectors)
    # find alpha
    a = find_alpha(Yi)
    # W(t + 1) = W(t)–a * [X(i)] ^ T *∆X(i) * [W`(t)] ^ T
    first_out_weight = find_first_weight(first_weight, a, vectors, delta_X, second_weight)
    # W`(t+1)=W`(t) – a`*[Y(i)]^T *∆X(i)
    second_out_weight = find_second_weight(second_weight, a, Yi, delta_X)

    return delta_X, Yi, X_i, delta_X, first_out_weight, second_out_weight


def process_compression_bytes_and_made_image(X_i, width, height, n, m, copy_pix, image, out_image_name: str) -> None:
    """Just make image from vectors of pixels"""
    # recover colors rgb
    colors = recovery_color(X_i)
    # convert to format pillow
    pixels_list = process_list_for_pillow(colors, width, height, n, m, copy_pix)
    recover = converter_to_format_pillow_from_matrix(pixels_list)
    make_image(image.mode, image.size, recover, out_image_name)


def learning_with_numpy(vectors, first_weight, second_weight) -> tuple:
    """Preparation step of learning, when we generated weights and get values from neural layer.
         :return tuple(delta_X, Yi, X_i, delta_X, first_weight, second_weight)"""
    Yi = numpy.dot(vectors, first_weight)
    # get X`(i)
    X_i = numpy.dot(Yi, second_weight)
    # get delta X`(i) - X(i)
    delta_X = numpy.subtract(X_i, vectors)
    # find alpha
    a = find_alpha(Yi)
    # W(t + 1) = W(t)–a * [X(i)] ^ T *∆X(i) * [W`(t)] ^ T
    first_out_weight = find_first_weight_with_numpy(first_weight, a, vectors, delta_X, second_weight)
    # W`(t+1)=W`(t) – a`*[Y(i)]^T *∆X(i)
    second_out_weight = find_second_weight_with_numpy(second_weight, a, Yi, delta_X)

    return delta_X, Yi, X_i, delta_X, first_out_weight, second_out_weight
