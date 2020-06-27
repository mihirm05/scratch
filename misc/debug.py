# debug practice


def method():
    outer_counter = 1
    inner_counter = 1

    while outer_counter <= 5:
        print(outer_counter, end="")
        while inner_counter <= 4:
            print(inner_counter, end="")
            inner_counter += 1
        inner_counter = 1
        outer_counter += 1
        print()


if __name__ == "__main__":
    method()
