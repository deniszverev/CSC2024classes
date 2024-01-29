# Develop two algorithms, one based on a loop structure and the other on a recursive structure, to print the daily
# salary of a worker who each day is paid twice the previous day's salary (starting with one penny for the first day's
# work) for a 30-day period.

def for_looping_algorithm(days):
    pay_list = [.01]

    for i in range(days):
        if i > 0:
            previous_day = pay_list[i-1]
            pay_list.append(previous_day * 2)

    print(len(pay_list))
    return pay_list


def recursion_algorithm(days):

    if days > 1:
        return recursion_algorithm(days - 1) * 2
    else:
        return .01


number_days = 30

print(for_looping_algorithm(number_days))
print(recursion_algorithm(number_days))
