# print pattern

# Input should be of the format 7 21
nums = input().split()


def processor():
    for i in range(int(nums[0])):
        if i < int(nums[0]) // 2:
            print(((int(nums[1]) - 3) // 2 - 3 * i) * '-' + (2 * i + 1) * '.|.' + ((int(nums[1]) - 3) // 2 - 3 * i) * '-')

        elif i == int(nums[0]) // 2:
            print(((int(nums[1]) - len('WELCOME')) // 2) * '-' + 'WELCOME' + ((int(nums[1]) - len('WELCOME')) // 2) * '-')

        elif i > int(nums[0]) // 2:
            i = int(nums[0]) - i - 1
            print(((int(nums[1]) - 3) // 2 - 3 * i) * '-' + (2 * i + 1) * '.|.' + ((int(nums[1]) - 3) // 2 - 3 * i) * '-')


if __name__ == "__main__":
    processor()
