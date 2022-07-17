import numpy as np
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
        
        
    
    def __init__(self, board):
        self.interpret_board(board)
        self.init_elements()
        self.init_sets()