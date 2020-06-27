# operator overloading (changing the operands while keeping
# operator same)


class Student:
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2

    # operator overloading
    def __add__(self, other):
        m1 = self.m1 + other.m1
        m2 = self.m2 + other.m2
        s = Student(m1, m2)
        return s

    # operator overloading
    def __sub__(self, other):
        m1 = self.m1 - other.m1
        m2 = self.m2 - other.m2
        sn = Student(m1, m2)
        return sn

    # operator overloading
    def __str__(self):
        return '{} {}'.format(self.m1, self.m2)


s1 = Student(50, 50)
s2 = Student(45, 55)

s3 = s1 + s2
s4 = s1 - s2

print(s3.m1, s3.m2)
print(s4.m1, s4.m2)

print(s1.__str__())
print(s2.__str__())



