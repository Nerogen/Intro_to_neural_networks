import random

from utils.config import left_max_weight, right_max_weight
from utils.util import transposition, find_Eq, open_and_process_image, user_input, make_cubes_and_vectors, learning, \
    process_compression_bytes_and_made_image, pickle_dump_to_file


def main() -> None:
    image_name, pixels, image, width, height = open_and_process_image()
    image_name = image_name.split("/")[-1].split(".")[0]
    print(f"Image: {image_name} | width: {width} | height: {height}")
    n, m, p, e = user_input(height)
    vectors, copy_pix = make_cubes_and_vectors(pixels, width, height, n, m)
    first_weight = [[random.uniform(left_max_weight, right_max_weight) for _ in range(p)] for _ in range(len(vectors[0]))]
    second_weight = transposition(first_weight)
    delta_X, Yi, X_i, delta_X, first_weight, second_weight = learning(vectors, first_weight, second_weight)
    count_rounds = 0
    while find_Eq(delta_X) > e:
        print(f'Round: {count_rounds} | Error: {find_Eq(delta_X)} > {e}')
        count_rounds += 1
        delta_X, Yi, X_i, delta_X, first_weight, second_weight = learning(vectors, first_weight, second_weight)

    out_image_name = "./directory_for_out_images/" + image_name + "_result"
    name_of_first_weight = "./directory_for_weights/" + image_name + "_w1.pickle"
    name_of_second_weight = "./directory_for_weights/" + image_name + "_w2.pickle"
    pickle_dump_to_file(name_of_first_weight, first_weight)
    pickle_dump_to_file(name_of_second_weight, second_weight)

    process_compression_bytes_and_made_image(X_i, width, height, n, m, copy_pix, image, out_image_name)
    print(f"Route of first weight of image: {name_of_first_weight}")
    print(f"Route of second weight of image: {name_of_second_weight}")
    print(f"Route of result image: {out_image_name}")


if __name__ == '__main__':
    main()
