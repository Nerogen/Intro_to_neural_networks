import numpy

from programm_modules.fast_method import learning_with_numpy
from utils.config import choose_menu_in_archive, error_incorrect_choose, choose_input, exit_massage
from utils.util import get_weights, open_and_process_image, user_input, \
    make_cubes_and_vectors, process_compression_bytes_and_made_image, pickle_dump_to_file, load_file, recovery_matrix


def main():
    flag = True
    while flag:
        print(choose_menu_in_archive)
        choose = int(input(choose_input))
        match choose:
            case 0:
                flag = False
                print(exit_massage)
            case 1:
                image_name, pixels, image, width, height = open_and_process_image()
                image_name = image_name.split("/")[-1].split(".")[0]
                print(f"Image: {image_name} | width: {width} | height: {height}")
                n, m, p, e = user_input(height)
                vectors, copy_pix = make_cubes_and_vectors(pixels, width, height, n, m)
                weight1, weight2 = get_weights("./directory_for_weights/" + image_name + "_w1.pickle",
                                               "./directory_for_weights/" + image_name + "_w2.pickle")
                vectors = numpy.array(vectors)
                weight1 = numpy.array(weight1)
                weight2 = numpy.array(weight2)
                delta_X, Yi, X_i, delta_X, weight1, weight2 = learning_with_numpy(vectors, weight1, weight2)
                file_name_for_compressed_bytes = "./directory_for_compressed_image_bytes/" + image_name + "_compressed.pickle"
                pickle_dump_to_file(file_name_for_compressed_bytes, (Yi, vectors, n, copy_pix))
                out_image_name = "./directory_for_compressed_image_bytes/" + image_name + "_result"
                process_compression_bytes_and_made_image(X_i, width, height, n, m, copy_pix, image, out_image_name)
                print(f"Route of compressed file: {file_name_for_compressed_bytes}")
                print(f"Route of compressed bytes: {out_image_name}")

            case 2:
                image_name, pixels, image, width, height = open_and_process_image()
                image_name = image_name.split("/")[-1].split(".")[0].split("_")[0]
                print(f"Image: {image_name} | width: {width} | height: {height}")
                weight1, weight2 = get_weights("./directory_for_weights/" + image_name + "_w1.pickle",
                                               "./directory_for_weights/" + image_name + "_w2.pickle")
                file_name_for_compressed_bytes = "./directory_for_compressed_image_bytes/" + image_name + "_compressed.pickle"
                delta_X, vectors, n, copy_pix = load_file(file_name_for_compressed_bytes)
                result = numpy.dot(delta_X, numpy.array(weight2))
                out_image_name = "./directory_for_recover_images/" + image_name + "_result"
                process_compression_bytes_and_made_image(result, width, height, n, n, copy_pix, image, out_image_name)
                print(f"Route of recover file: {out_image_name}")
            case _:
                print(error_incorrect_choose)


if __name__ == '__main__':
    main()
