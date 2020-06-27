# selection sort


def sort(num):
    """
    :param num: an unsorted list
    :return: None
    """
    for i in range(5):
        val = i
        for j in range(i, 6):
            if num[j] < num[val]:
                val = j

        temp = num[i]
        num[i] = num[val]
        num[val] = temp
        print(num)


if __name__ == "__main__":
    nums = [5, 3, 8, 6, 7, 2]
    print(nums)
    sort(nums)
