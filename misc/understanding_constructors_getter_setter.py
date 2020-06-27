# constructors, getters and setters


class Student:

    school = 'school'

    # constructors
    def __init__(self, m1, m2, m3):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    # getters
    def get_m1(self):
        return self.m1

    def get_m2(self):
        return self.m2

    def get_m3(self):
        return self.m3

    # setters
    def set_m1(self, val):
        self.m1 = val

    def set_m2(self, val):
        self.m2 = val

    def set_m3(self, val):
        self.m3 = val

    def avg(self):
        return (self.m1 + self.m2 + self.m3) / 3

    @classmethod
    def get_school_name(cls):
        return cls.school

    @staticmethod
    def info():
        return 'This is the student class and this is a static method'


s1 = Student(12, 12, 21)
s2 = Student(32, 14, 10)

# testing getters
print('original values')
print('Subject 1 marks for students are: ', s1.get_m1(), ' and ', s2.get_m1())
print('Subject 2 marks for students are: ', s1.get_m2(), ' and ', s2.get_m2())
print('Subject 3 marks for students are: ', s1.get_m3(), ' and ', s2.get_m3())

# testing setters
s1.set_m1(14)
s1.set_m2(56)
s1.set_m3(83)

s2.set_m1(42)
s2.set_m2(51)
s2.set_m3(8)

print('new values')
print('Subject 1 marks for students are: ', s1.get_m1(), ' and ', s2.get_m1())
print('Subject 2 marks for students are: ', s1.get_m2(), ' and ', s2.get_m2())
print('Subject 3 marks for students are: ', s1.get_m3(), ' and ', s2.get_m3())

print(s1.get_school_name())
print(s2.get_school_name())
print(Student.get_school_name())

print(Student.info())






