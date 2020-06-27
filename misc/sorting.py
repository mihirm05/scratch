# Enter your code here. Read input from STDIN. Print output to STDOUT
# Example input : Sorting1234
# Example output : ginortS1324
a = input()
a = list(a)

even = []
odd = []
uppr = []
lwr = []


def processor():
    for i in a:
        # print(i)
        if i.islower():
            lwr.append(i)
        elif i.isupper():
            uppr.append(i)
        elif int(i) % 2 == 0:
            even.append(i)
        elif int(i) % 2 != 0:
            odd.append(i)

    output = sorted(lwr) + sorted(uppr) + sorted(odd) + sorted(even)
    # print(sorted(lwr))
    # print(sorted(uppr))
    # print(sorted(odd))
    # print(sorted(even))
    output = ''.join(output)
    print(output)


if __name__ == "__main__":
    processor()
