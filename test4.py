number = input().split()


def processor():
    names = []
    grades = []
    results = []

    for i in range(0, 2 * int(number[0])):
        if i % 2 == 0:
            names.append(input().split())
        else:
            grades.append(input().split())

    backup_grades = [float(item) for sublist in grades for item in sublist]
    grade_set = set(backup_grades)
    second_least = sorted(grade_set)[1]

    combined_list = [grades[i] + names[i] for i in range(len(names))]
    combined_list.sort()
    # print(combined_list)

    for k in range(len(grades)):
        if float(combined_list[k][0]) == second_least:
            results.append(combined_list[k])
    results.sort(key=lambda x: x[1])

    for m in range(len(results)):
        print(results[m][1])


if __name__ == "__main__":
    processor()
