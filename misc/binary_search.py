# Binary Search
# concept of lower and upper bound used. mid is used to
# divide the list in two parts. if the desired value
# is bigger than the value at mid, then update the lower
# bound or else update the upper bound


location = 0


def search(lists, n):
    lower = 0
    upper = len(lis) - 1

    while lower <= upper:
        mid = int((lower + upper) / 2)

        if lis[mid] == n:
            globals()['location'] = mid + 1
            return True
        else:
            if lis[mid] < n:
                lower = mid + 1
            elif lis[mid] > n:
                upper = mid - 1
    return False


lis = [1, 2, 3, 4, 5, 6, 7, 8, 9]

if search(lis, 6):
    print('Found at ', location)
else:
    print('Not Found')
