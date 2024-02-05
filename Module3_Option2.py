# Write two Python functions to find the minimum number in a list. The first function should compare each number to
# every other number on the list O(n2). The second function should be linear O(n).
import random


def quadratic_search(arr):
    min_number = arr[0]
    length = len(arr)

    for i in range(length):
        for j in range(len(arr)):
            if arr[j] < min_number:
                min_number = arr[j]

    return min_number


def linear_search(arr):
    min_number = arr[0]
    length = len(arr)

    for i in range(length):
        if arr[i] < min_number:
            min_number = arr[i]

    return min_number


random_array = [random.randint(1, 100) for _ in range(1000)]

print(quadratic_search(random_array))
print(linear_search(random_array))

