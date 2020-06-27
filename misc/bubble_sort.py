# Bubble sort
# iterate over the entire list, swap elements if bigger
# element is on the left. once this passage is completed
# we get the largest element to the right of the list
# problem: swapping consumes processing power


import random


def sort(lists):
    for i in range(len(lists) - 1, 0, -1):
        for j in range(i):
            if nums[j] > nums[j + 1]:
                temp = nums[j]
                nums[j] = nums[j + 1]
                nums[j + 1] = temp
        print('intermediate: ',nums)


if __name__ == "__main__":
    nums = random.sample(range(1, 1000), 5)
    print('unsorted: ', nums)
    sort(nums)
    print('sorted: ', nums)
