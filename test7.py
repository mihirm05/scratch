# incomplete code


import string

a = input()
alphabet = string.ascii_lowercase
l = []


def processor():
    for i in range(int(a)):
        # print(alpha[i:n])
        s = '-'.join(alpha[i:int(a)])
        print(s)
        # l.append((s[::-1] + s[1:]).center(4 * n - 3, '-'))

    # print(l)
    # for i in range(2 * int(a) + 1):
    #    if i <= (int(a) - 1):
    #        print(2 * (int(a) - i - 1) * '-' + alphabet[int(a) - i - 1] + 2 * (int(a) - i - 1) * '-')
    #    elif i == int(a):
    #        print(alphabet[int(a) - 1::-1] + '-' + alphabet[:int(a)])
    #    elif i >= (int(a) + 1):
    #        i = int(a) - i - 1
    #        print(2 * (int(a) - i - 1) * '-' + alphabet[int(a) - i - 1] + 2 * (int(a) - i - 1) * '-')


if __name__ == "__main__":
    processor()
