# Inner class


class Student:
    def __init__(self, name='no name', roll=0):
        self.name = name
        self.roll = roll
        # self.lap = self.Laptop('HP', 'i5', 4)
        # self.lap.show()

    def show(self):
        print('Outer class show: ',
              self.name, self.roll)

    class Laptop:
        def __init__(self, brand='HP', cpu='i5', ram=4):
            self.brand = brand
            self.cpu = cpu
            self.ram = ram

        def show(self):
            print('Inner class show: ', self.brand,
                  self.cpu, self.ram)

        # creating an inner class object:
        # (1) self.lap = self.Laptop()   [inside Student init]
        # (2) lap1 = Student.Laptop()  [in the main code]


s1 = Student('student 1 ', 2)
s1.show()
lap1 = s1.Laptop('HP', 'i5', 8)
lap2 = Student.Laptop('HP', 'i5', 8)
lap1.show()
lap2.show()

print()

s2 = Student('student 2 ', 4)
s2.show()
lap3 = s2.Laptop()
lap4 = Student.Laptop()
lap3.show()
lap4.show()

print()

print(id(lap1))
print(id(lap2))
print(id(lap3))
print(id(lap4))


