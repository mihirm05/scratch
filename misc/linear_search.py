# Linear Search
# take each element and compare it with the desired value
# if not equal move on, if equal print found


location = 0


def search(lists, n):
    print(lis)
    i = 0

    while i < len(lis):
        if lis[i] == n:
            # print(list[i])
            globals()['location'] = i + 1
            # print('Found at ', i)
            return True
        i += 1
    return False


if __name__ == "__main__":
    lis = [1, 2, 3, 4, 5, 6, 7]
    if search(lis, 4):
        print('Found at ', location)
    else:
        print('Not Found')
