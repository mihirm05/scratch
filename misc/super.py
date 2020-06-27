# super()


class A:
    def __init__(self):
        print('Init A')

    def feature_1(self):
        print('feature 1 A')


class B:
    def __init__(self):
        print('Init B')

    def feature_1(self):
        print('feature 1 B')


class C(B, A):

    # Method resolution order is Left to Right
    # if the order of inheritance was (A, B) then the output
    # would be 'A' in place of 'B' everywhere
    def __init__(self):
        super().__init__()
        print('Init C')


a = C()
a.feature_1()
