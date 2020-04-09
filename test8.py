# basic

import sys


def processor():
    a = int(input('enter val 1: ')[0])
    b = int(input('enter val 2: ')[0])
    return a + b


if __name__ == "__main__":
    results = processor()
    print(results)
