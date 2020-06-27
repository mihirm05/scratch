def divider(value):
    """
    :param: total number of divisions desired
    :return: None
    """

    for i in range(value):
        try:
            numerator, denominator = map(int, input().split())
            print(numerator // denominator)
        except ZeroDivisionError as E:
            print('Error Code: ', E)
        except ValueError as F:
            print('Error Code: ', F)
        finally:
            print('Exit')


if __name__ == "__main__":
    total_divisions = int(input('Enter number of division operations '))
    divider(total_divisions)