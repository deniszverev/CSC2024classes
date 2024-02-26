# You'll build a simple binary search tree in this activity.
# Build a Node class. It should have attributes for the data it stores as well as its left and right children.
# As a bonus, try including the Comparable module and make nodes compare using their data attribute. (Ruby??)
# Build a Tree class that accepts an array when initialized.
# The Tree class should have a root attribute that uses the return value of #build_tree which you'll write next.
# Write a #build_tree method that takes an array of data (e.g. [1, 7, 4, 23, 8, 9, 4, 3, 5, 7, 9, 67, 6345, 324]) and
# turns it into a balanced binary tree full of Node objects appropriately placed (don't forget to sort and remove
# duplicates!). The #build_tree method should return the level-1 root node.
# Write an #insert and #delete method which accepts a value to insert/delete.S

class Node:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


class Tree:
    def __init__(self, data):
        # Sort and remove duplicates
        self.root = self.build_tree(sorted(set(data)))

    def build_tree(self, data):
        if not data:
            return None
        # Binary tree
        mid = len(data) // 2
        root = Node(data[mid])
        root.left = self.build_tree(data[:mid])
        root.right = self.build_tree(data[mid + 1:])
        return root

    def insert(self, value):
        # Accepts a value to insert
        self.root = self._insert(self.root, value)

    def delete(self, value):
        # Accepts a value to delete
        self.root = self._delete(self.root, value)

    def _insert(self, node, value):
        # Helper method for insert handles three cases for inserting a node
        if node is None:
            return Node(value)
        if value < node.data:
            node.left = self._insert(node.left, value)
        elif value > node.data:
            node.right = self._insert(node.right, value)
        return node

    def _delete(self, node, value):
        # Helper method for delete handles three cases for deleting a node
        if node is None:
            return node
        if value < node.data:
            node.left = self._delete(node.left, value)
        elif value > node.data:
            node.right = self._delete(node.right, value)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            node.data = self.find_min(node.right).data
            node.right = self._delete(node.right, node.data)
        return node

    def find_min(self, node):
        # Helper method to find minimum value node in a subtree
        current = node
        while current.left:
            current = current.left
        return current


def print_tree(node, level=0):
    if node is not None:
        print_tree(node.left, level + 1)
        print(' ' * 6 * level + '-> ' + str(node.data))
        print_tree(node.right, level + 1)


input_array = [1, 7, 4, 23, 8, 9, 4, 3, 5, 7, 9, 67, 6345, 324]
tree = Tree(input_array)
print_tree(tree.root)
print("="*30)
tree.insert(10)
print_tree(tree.root)
print("="*30)
tree.delete(10)
print_tree(tree.root)