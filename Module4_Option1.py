# Design and implement an experiment that will compare the performance of the Python list based stack and queue with
# the linked list implementation. Provide a brief discussion of both stacks and queues for this activity.
import time
import statistics
import matplotlib.pyplot as plt


class ListStack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class ListQueue:
    def __init__(self):
        self.head = None
        self.last = None

    def enqueue(self, data):
        if self.last is None:
            self.head = Node(data)
            self.last = self.head
        else:
            self.last.next = Node(data)
            self.last = self.last.next

    def dequeue(self):
        if self.head is None:
            print("Queue is empty.")
            return None
        else:
            to_return = self.head.data
            self.head = self.head.next
            return to_return

    def front(self):
        return self.head


def measure_time(structure, operation, size):

    start = time.time()

    if operation == "push" or operation == "enqueue":
        for i in range(size):
            getattr(structure, operation)(i)
    elif operation == "pop" or operation == "dequeue":
        for i in range(size):
            getattr(structure, operation)()
    else:
        for i in range(size):
            getattr(structure, operation)

    finish = time.time()

    return finish-start


sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 2000000]
push_output_times = []
pop_output_times = []
peek_output_times = []

for size in sizes:
    list_stack = ListStack()
    list_queue = ListQueue()

    # Push/Enqueue
    time_push_stack = measure_time(list_stack, "push", size)
    time_enqueue_queue = measure_time(list_queue, "enqueue", size)
    push_output_times.append(time_enqueue_queue - time_push_stack)

    # Pop/Dequeue
    time_pop_stack = measure_time(list_stack, "pop", size)
    time_dequeue_queue = measure_time(list_queue, "dequeue", size)
    pop_output_times.append(time_dequeue_queue - time_pop_stack)

    # Access/Peek/Front
    time_peek_stack = measure_time(list_stack, "peek", size)
    time_front_queue = measure_time(list_queue, "front", size)
    peek_output_times.append(time_front_queue - time_peek_stack)

print(statistics.mean(push_output_times))
print(statistics.mean(pop_output_times))
print(statistics.mean(peek_output_times))


plt.plot(sizes, push_output_times)
plt.xlabel("Iterations")
plt.ylabel("Push Operation Difference(sec)")
plt.show()

plt.plot(sizes, pop_output_times)
plt.xlabel("Iterations")
plt.ylabel("Pop Operation Difference(sec)")
plt.show()

plt.plot(sizes, peek_output_times)
plt.xlabel("Iterations")
plt.ylabel("Access Operation Difference(sec)")
plt.show()
