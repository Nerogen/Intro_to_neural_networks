from programm_modules import archive, fast_method, slow_method
from utils.config import manager_menu, choose_input, error_incorrect_choose, exit_massage


def main() -> None:
    flag = True
    while flag:
        print(manager_menu)
        choose = int(input(choose_input))
        match choose:
            case 0:
                flag = False
                print(exit_massage)
            case 1:
                fast_method.main()
            case 2:
                slow_method.main()
            case 3:
                archive.main()
            case _:
                print(error_incorrect_choose)


if __name__ == '__main__':
    main()
