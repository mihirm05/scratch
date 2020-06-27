# method overloading : same method name but type and number
# of arguments vary 


class Student:
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2

    def sum(self, *a):
        s = sum(a)
        return s


s1 = Student(21, 12)
print(s1.sum(5, 6, 1, 5))