# Example input
# 3 1000
# 2 5 4
# 3 7 8 9
# 5 5 7 8 9 10

# Example output
# 206

import itertools

l = []
ul = []
sum_vals = []
sum_vals_updated = []


def processor():
    lists, divider = map(int, input().split())
    for _ in range(lists):
        l.append(list(map(int, input().split())))

    for item in l:
        for i in range(1, len(item)):
            item[i] = (item[i] ** 2)
        ul.append(item)

    for values in ul:
        values.pop(0)
    print(ul)
    print(list(itertools.product(*ul)))
    for triplets in list(itertools.product(*ul)):
        # print(sum(triplets))
        sum_vals.append(sum(triplets))
    print(sum_vals)
    sum_vals_updated = [summation % divider for summation in sum_vals]
    print(sum_vals_updated)
    # for iter in sum_vals:
    #    sum_vals.append(iter%divider)
    print(max(sum_vals_updated))


if __name__ == "__main__":
    processor()
