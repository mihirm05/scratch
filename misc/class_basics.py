# defining classes, creating objects and comparing objects


class Computer:

    def __init__(self):
        print('New object created')
        self.name = 'nike'
        self.age = 20

    def update(self, name, age):
        self.name = name
        self.age = age

    def compare(self, other):
        if self.name == other.name and self.age == other.age:
            print('Same name and age')
            print()

        elif self.name == other.name and self.age != other.age:
            print('Same name but different age')
            print()

        elif self.name != other.name and self.age == other.age:
            print('Different name but same age')
            print()

        else:
            print('Different name and age')
            print()


if __name__ == "__main__":
    c1 = Computer()
    c2 = Computer()

    c1.compare(c2)

    c1.update(name=c1.name, age=4)
    print('age in c1 updated')

    c1.compare(c2)

    c2.update(name='adidas', age=c2.age)
    print('name in c2 updated')

    c1.compare(c2)
