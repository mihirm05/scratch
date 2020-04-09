# Example input
# 5 3
# 89 90 78 93 80
# 90 91 85 88 86
# 91 92 83 89 90.5

# Example Output
# 90.0
# 91.0
# 82.0
# 90.0
# 85.5

a, b = input().split()
val = []
avg = []


def processor():
    for _ in range(int(b)):
        val.append(list(map(float, input().split())))

    counter = 0
    result = []

    for i in range(int(a)):
        summer = 0
        for values in val:
            summer = summer + values[i]
            counter = counter + 1
        avg.append(summer)

    result = [iterator / int(b) for iterator in avg]

    result = ['%.2f' % elem for elem in result]
    for values in result:
        print(float(values))


if __name__ == "__main__":
    processor()




