# decorators


def div(a, b):
    print(a / b)


def smart_div(func):
    def inner(a, b):
        print(a)
        print(b)
        if a < b:
            a, b = b, a
            return func(a, b)

        else:
            return func(a, b)
    return inner


if __name__ == "__main__":
    div = smart_div(div)
    div(2, 4)

