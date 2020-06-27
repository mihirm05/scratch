# iterators
# values would be a custom iterator, every time the for
# loop runs, the __next__() is invoked and the value of
# num increments. 10 is the limiting condition. power of
# iterator, even after specifically printing the
#  __next__() twice, when the loop begins, the values
# printed are from 3 rather than 1


class TopTen:
    def __init__(self):
        self.num = 1

    def __iter__(self):
        print('iter')
        return self

    def __next__(self):
        if self.num <= 10:
            print('inside if')
            val = self.num
            self.num += 1
            return val

        else:
            raise StopIteration


if __name__ == '__main__':
    values = TopTen()
    print(values.__next__())
    print(values.__next__())
    for i in values:
        print(i)