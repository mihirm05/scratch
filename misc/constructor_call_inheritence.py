# constructor calls for inheritance


class A:
    def __init__(self):
        print('Init A')


class B(A):
    def __init__(self):
        super().__init__()
        print('Init B')


if __name__ == "__main__":
    a = B()
