# Exception Handling, critical statements to be written
# in try, run time errors to be handled in except.
# Instructions to be executed surely to be typed in finally


def function(a, b):
    try:
        a = int(a)
        b = int(b)

        print('Connection opened')
        print(a / b)

    except ZeroDivisionError as e:
        print('Can not divide by 0', e)

    except ValueError as e:
        print(e)

    finally:
        print('Connection closed')


if __name__ == "__main__":
    val1 = input('Enter the first value ')
    val2 = input('Enter the second value ')
    function(val1, val2)
