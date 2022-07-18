import numpy as np
import itertools
"""
python environment for sudoku
"""

"""
Wrapper for position on a sudoku board. A vector [0,1]^9 representing
which digits can be in that element.
"""
class Element:
    
    def __init__(self, el):
        self.el = el
    
    def is_solved(self):
        return np.sum(self.el) == 1
    
    def is_conflict(self):
        return np.sum(self.el) == 0
    
    # number of possible digits
    def confusion(self):
        return np.sum(self.el)

class Sudoku:
    
    """
    board comes in as a 9x9 [row, coln] array of ints. interpret into 9x9x9 
    boolean. 
    """
    def interpret_board(self, board):
        self.notes = np.ones((9,9,9), dtype=bool)
        for i in range(1,10):
            for j in range(1,10):
                if(not board[i][j] is None):
                    digit = board[i][j]
                    self.notes[i][j] = np.array([False if i!=digit else True for i in range(1,10)])
    
    def init_elements(self):
        self.elements = []
        for i in range(9):
            for j in range(9):
                self.elements.append(self.notes[i,j])
    
    def init_sets(self):
        indices = np.array(range(81)).reshape(9,9)
        rows = [set([self.elements[index]]) for index in indices]
        cols = [set([self.elements[index]]) for index in indices.T]
        box_indices = [indices[i:i+3,j:j+3] for i, j in itertools.product(range(0, 10, 3), range(0, 10, 3))]
        boxs = [set[box_index] for box_index in box_indices]
        self.rows = rows
        self.cols = cols
        self.boxs = boxs
        self.sets = rows + cols + boxs
    
    def __init__(self, board, verbose=0):
        self.interpret_board(board)
        self.init_elements()
        self.init_sets()
    
    """ returns the number of permutations possible given notes. """
    def naive_size(self):
        return np.prod([el.confusion() for el in self.elements])
    
    def num_solved(self):
        return np.sum([el.is_solved() for el in self.elements])
    
    """ performs all logic """ 
    def logic(self):
        return None
    
    def solve(self):
        repeat = True
        while(repeat):    
            prev_notes = self.notes
            self.logic()
            repeat = np.array_equiv(prev_notes, notes)
            if(verbose):
                print("Iterated logic.")
                print("Puzzle size: %d"%self.naive_size())
                print("Solved elements: %d"%self.num_solved())
            

if __name__ == "__main__":
    