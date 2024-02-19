# Illustrate the linear probing method in hashing. Explain its performance analysis. Lastly, discuss how rehashing
# overcomes the drawbacks of linear probing. Provide at least one visual in your activity.
import time
import statistics
import matplotlib.pyplot as plt


class LinearProbingHashTable:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        # Check if empty
        while self.table[index] is not None:
            # Linearly increase index
            index = (index + 1) % self.size

        self.table[index] = (key, value)

    def search(self, key):
        index = self.hash_function(key)

        while self.table[index] is not None:
            # Get key in the table
            stored_key, value = self.table[index]
            # Compare key to searched key
            if stored_key == key:
                return value
            index = (index + 1) % self.size

        raise KeyError(f"Key {key} not found.")

    def delete(self, key):
        index = self.hash_function(key)
        # Similar to search()
        while self.table[index] is not None:
            stored_key, value = self.table[index]
            if stored_key == key:
                # Delete Operation
                self.table[index] = None
                return
            index = (index + 1) % self.size

        raise KeyError(f"Key {key} not found.")


class RehashingHashTable:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size

    def hash_function(self, key):
        return hash(key) % self.size

    def secondary_hash_function(self, key):
        return hash(key + 1) % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        offset = self.secondary_hash_function(key)

        while self.table[index] is not None:
            index = (index + offset) % self.size

        self.table[index] = (key, value)

    def search(self, key):
        index = self.hash_function(key)
        offset = self.secondary_hash_function(key)

        while self.table[index] is not None:
            stored_key, value = self.table[index]
            if stored_key == key:
                return value
            index = (index + offset) % self.size

        raise KeyError(f"Key {key} not found.")

    def delete(self, key):
        index = self.hash_function(key)
        offset = self.secondary_hash_function(key)

        while self.table[index] is not None:
            stored_key, value = self.table[index]
            if stored_key == key:
                self.table[index] = None
                return
            index = (index + offset) % self.size

        raise KeyError(f"Key {key} not found.")


def measure_time(structure, operation, size):

    start = time.time()

    if operation == "insert":
        for i in range(size):
            getattr(structure, operation)(i % 10, i)
    else:
        for i in range(size):
            getattr(structure, operation)(i)

    finish = time.time()

    return finish-start


table_size = 10000
iterations = table_size * 3 // 4

performance_chart = []

for i in range(iterations):
    linear_probing_table = LinearProbingHashTable(table_size)
    rehashing_table = RehashingHashTable(table_size)

    time_insert1 = measure_time(linear_probing_table, "insert", i)
    time_insert2 = measure_time(rehashing_table, "insert", i)
    time_insert = (max(0, time_insert2 - time_insert1)) / time_insert1 if time_insert1 != 0 else 0

    if time_insert > 0:
        performance_chart.append(time_insert)

print(statistics.mean(performance_chart))
plt.plot(performance_chart)
plt.xlabel("Iterations")
plt.ylabel("Operation Difference(sec)")
plt.show()
