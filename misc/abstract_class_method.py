# abstract class and method


from abc import ABC, abstractmethod


# abstract class
class Computer:
    @abstractmethod
    def process(self):
        pass


# regular class
class Laptop(Computer):
    def process(self):
        print("runs")


lap1 = Laptop()
lap1.process()
