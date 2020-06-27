# generator
# benefit of using generator
# (i) no need to define __iter__() and __next__()
# (ii) suppose we want to work with 1000 records from a
# dataset, instead of loading every record in the memory,
# generator facilitates working with a single record at
# a time and reducing the load on memory.


def top_ten():
    n = 1

    while n <= 10:
        sq = n**2
        yield sq
        n += 1


values = top_ten()

for i in values:
    # print(id(i))
    print(i)