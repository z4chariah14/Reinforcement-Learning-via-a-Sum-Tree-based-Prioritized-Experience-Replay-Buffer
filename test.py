from src.sum_tree import SumTree

tree = SumTree(5)
tree.add(10, "A")
tree.add(20, "B")
tree.add(30, "C") 

print(tree.totalPriority()) # Should be 60
print(tree.get_leaf(5))      # Should return index, 10, "A"
print(tree.get_leaf(25))     # Should return index, 20, "B"