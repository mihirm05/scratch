# inheritance


class A:
    def __init__(self):
        print('Object of A created')

    def feature_1(self):
        print('feature 1 ')


class B:
    def __init__(self):
        print('Object of B created')

    def feature_2(self):
        print('feature 2')


class C(A, B):
    def __init__(self):
        print('Object of C created')

    def feature_3(self):
        print('feature 3')


a = A()
a.feature_1()

b = B()
b.feature_2()

c = C()
c.feature_3()
