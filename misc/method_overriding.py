# method overriding : same name and same number and type of
# arguments but different functioning


class A:
    def show(self):
        print('In A')


class B(A):
    def show(self):
        print('In B')

# show() of B will be invoked if it is present
# if not then the show() of A will be invoked


a = A()
a.show()
b = B()
b.show()
