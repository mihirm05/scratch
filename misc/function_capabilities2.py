# function assignment to a variable 
def test():
    funcVar = abs 
    print(funcVar(-2))


# object as functions 
class Print:
    def __init__(self,s):
        self.string = s 


    def __call__(self):
        print(self.string)


if __name__ == "__main__":
    test()
    s1 = Print('Hello')
    s1()




