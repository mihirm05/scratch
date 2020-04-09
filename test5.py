# wrapping text

string = input()
num = input()


def processor():
    counter = 0
    for i in range(len(string)):
        if counter % int(num) == 0:
            print(string[i:i + int(num)])
        counter += 1


if __name__ == "__main__":
    processor()
