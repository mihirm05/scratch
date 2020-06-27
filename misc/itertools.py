from itertools import product


def iterators(first, second):
    """
    :param: first set of values
    :param: second set of values
    :return: None
    """

    # different print options available in iterators
    print(list(product(first, second)))
    print(*product(first, second))


if __name__ == "__main__":
    first_input = input('Enter the first set ').split()
    first_input = [int(i) for i in first_input]
    second_input = input('Enter the second set ').split()
    second_input = [int(i) for i in second_input]
    iterators(first_input, second_input)


