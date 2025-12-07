import numpy as np

class SumTree():
    def __init__(self, N):
        self.Tree = np.zeros(2*N-1)
        self.capacity = N
        self.data = np.zeros(N, dtype = object)
        self.data_pointer = 0
        
        
    def totalPriority(self):
        return self.Tree[0]
    
    def update(self, tree_index, priority):
        change = priority - self.Tree[tree_index]
        self.Tree[tree_index] = priority
        i = tree_index
        while i != 0:
            self.Tree[(i-1)//2] += change
            i = (i - 1) // 2
        
    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.data_pointer += 1
        if self.data_pointer == self.capacity:
            self.data_pointer = 0
        self.update(tree_index, priority)
        
    def get_leaf(self, value):
        i = 0
        while i < self.capacity - 1:
            left = 2*i +1
            right = 2*i +2
            if self.Tree[left] > value:
                i = left
            else:
                i = right
                value = value - self.Tree[left] 
        data_index = i - (self.capacity - 1)
        return i, self.Tree[i], self.data[data_index]