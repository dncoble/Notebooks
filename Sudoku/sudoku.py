import numpy as np
import itertools
"""
python environment for sudoku
"""

"""
position on a sudoku board. A vector [0,1]^9 representing
which digits can be in that element.
"""
class Cell:
    
    def __init__(self, el):
        self.el = el
    
    def __call__(self):
        return self.el
    
    def __getitem__(self, key):
        return self.el[key]
    
    def is_solved(self):
        return np.sum(self.el) == 1
    
    def is_conflict(self):
        return np.sum(self.el) == 0
    
    # number of possible digits
    def confusion(self):
        return np.sum(self.el)
    
    def possible_nums(self):
        return np.array([1,2,3,4,5,6,7,8,9])[self.el]

"""
A set of cells
"""
class CellSet:
    
    """
    cells is a numpy array of Cell objects
    """
    def __init__(self, cells):
        self.cells = cells
    
    def __call(self):
        return self.cells
    
    def __getitem__(self, key):
        return self.cells[key]
    
    def __len__(self):
        return self.cells.size
    """
    return all cells in self that are also in other
    """
    def intersection(self, other):
        rtrn = np.array([el in other.cells for el in self.cells])
        return CellSet(self.cells[rtrn])
    
    """
    return all cells in self or other
    """
    def union(self, other):
        s = self.subtract(other)
        return CellSet(np.append(self.cells, s))
    
    """
    return all cells in self that are not in other
    """
    def subtract(self, other):
        rtrn = np.array([not el in other.cells for el in self.cells], dtype=np.bool)
        return CellSet(self.cells[rtrn])
    
    """
    if the number n only exists in the subset of self
    """
    def must_in(self, n, subset):
        nset = self.subtract(subset)
        return not any([c[n-1] for c in nset.cells])
    
    """
    eliminate the number n as a possibility in all elems. return True
    if any True were set to False
    """
    def elim_n(self, n):
        rtrn = False
        for cell in self.cells:
            rtrn = rtrn or cell[n-1]
            cell.el[n-1] = False
        return rtrn
    
    """
    all numbers which appear anywhere as True
    """
    def all_nums(self):
        rtrn = np.zeros((9), dtype=np.bool)
        for cell in self.cells:
            rtrn = np.logical_or(rtrn, cell())
        return np.array([1,2,3,4,5,6,7,8,9])[rtrn]
    
    """
    get all cells where n is True
    """
    def support(self, n):
        rtrn = np.array([cell[n-1] for cell in self.cells])
        return CellSet(self.cells[rtrn])
    
class Sudoku:
    
    """
    board comes in as a 9x9 [row, coln] array of ints. interpret into 9x9x9 
    boolean. 
    """
    def interpret_board(self, board):
        self.notes = np.ones((9,9,9), dtype=bool)
        for i in range(0,9):
            for j in range(0,9):
                if(not board[i][j] is None):
                    digit = board[i][j]
                    self.notes[i][j] = np.array([False if i!=digit else True for i in range(1,10)])
    
    def init_cells(self):
        self.cells = []
        for i in range(9):
            for j in range(9):
                self.cells.append(Cell(self.notes[i,j]))
    
    def init_multiedges(self):
        indices = np.array(range(81), dtype=np.int).reshape(9,9)
        # I have a problem with list comprehension
        rows = [CellSet(np.array([self.cells[index] for index in index_set])) for index_set in indices]
        cols = [CellSet(np.array([self.cells[index] for index in index_set])) for index_set in indices.T]
        box_indices = np.array([indices[i:i+3,j:j+3].flatten() for i, j in itertools.product(range(0, 9, 3), range(0, 9, 3))])
        boxs = [CellSet(np.array([self.cells[index] for index in index_set])) for index_set in box_indices]
        self.rows = rows
        self.cols = cols
        self.boxs = boxs
    
    def __init__(self, board, verbose=0):
        self.verbose=verbose
        self.interpret_board(board)
        self.init_cells()
        self.init_multiedges()
    
    def solved_string(self):
        rtrn = ' ___________________________________ \n'
        digits = np.array([1,2,3,4,5,6,7,8,9])
        nums = np.array([digits[c.el] for c in self.cells]).reshape((9,9))
        for n in nums:
            rtrn += '|   |   |   |   |   |   |   |   |   |\n'
            rtrn += '| %d | %d | %d | %d | %d | %d | %d | %d | %d |\n'%tuple(n)
            rtrn += '|___|___|___|___|___|___|___|___|___|\n'
        return rtrn
    
    def notes_string(self):
        notes_encode = np.array([' ' for i in range(729)], dtype=str).reshape((27,27))
        c = 0
        for i, j, k in itertools.product(range(9), range(9), range(9)):
            if(self.cells[9*i+j][k]):
                x = 3*i+k//3
                y = 3*j+k%3
                notes_encode[x, y] = str(k+1)
                c += 1
        print(notes_encode)
        rtrn = ' _____________________________________________________ \n'
        notes_encode = notes_encode.flatten()
        m = 0
        for i in range(9):
            rtrn += '|     |     |     |     |     |     |     |     |     |\n'
            for j in range(3):
                for k in range(9):
                    rtrn += '| '
                    for l in range(3):
                        rtrn += notes_encode[m]; m += 1
                    rtrn += ' '
                rtrn += '|\n'
            rtrn += '|_____|_____|_____|_____|_____|_____|_____|_____|_____|\n'
        return rtrn
        
        
    def __str__(self):
        for cell in self.cells:
            if(cell.confusion() != 1):
                return self.notes_string()
        return self.solved_string()
    
    """ returns the number of permutations possible given notes. """
    def naive_size(self):
        return np.prod([el.confusion() for el in self.cells])
    
    def num_solved(self):
        return np.sum([el.is_solved() for el in self.cells])
    
    """
    exclusive logic between the edges in set1 and set2
    """
    def exclusive_logic(self, set1, set2):
        logiced = False
        for s1 in set1:
            for s2 in set2:
                intersect = s1.intersection(s2)
                p = intersect.all_nums()
                for n in p:
                    if(s1.must_in(n, intersect)):
                        u = s2.subtract(intersect)
                        logiced = u.elim_n(n) or logiced
                    if(s2.must_in(n, intersect)):
                        u = s1.subtract(intersect)
                        logiced = u.elim_n(n) or logiced
        return logiced
    
    """
    inclusive logic within the multiedge set
    """
    def inclusive_logic(self, multiedge):
        logiced = False
        for i in range(1, 9):
            for subset in itertools.combinations(multiedge, i):
                subset = CellSet(np.array(subset))
                n_subset = multiedge.subtract(subset)
                digits = subset.all_nums()
                if(len(digits) == len(subset)):
                    for d in digits:
                        logiced = n_subset.elim_n(d) or logiced
        return logiced
    
    """ performs all logic """ 
    def logic(self):
        logiced = False # flag if any logic reduction was performed
        
        # exclusive logic
        logiced = self.exclusive_logic(self.rows, self.cols) or logiced 
        logiced = self.exclusive_logic(self.rows[0:3], self.boxs[0:3]) or logiced
        logiced = self.exclusive_logic(self.rows[3:6], self.boxs[3:6]) or logiced
        logiced = self.exclusive_logic(self.rows[6:9], self.boxs[6:9]) or logiced
        logiced = self.exclusive_logic(self.cols[0:3], [self.boxs[0],self.boxs[3],self.boxs[6]]) or logiced
        logiced = self.exclusive_logic(self.cols[3:6], [self.boxs[1],self.boxs[4],self.boxs[7]]) or logiced
        logiced = self.exclusive_logic(self.cols[6:9], [self.boxs[2],self.boxs[5],self.boxs[8]]) or logiced
       
        # inclusive logic
        for row in self.rows:
            logiced = self.inclusive_logic(row) or logiced
        for col in self.cols:
            logiced = self.inclusive_logic(col) or logiced
        for box in self.boxs:
            logiced = self.inclusive_logic(box) or logiced
        
        # check for contradictions
        for cell in self.cells:
            if(cell.is_conflict()):
                return -1
        
        return logiced
                
    
    def solve(self):
        repeat = True
        while(repeat):    
            repeat = self.logic()
            if(repeat == -1):
                if(self.verbose):
                    print("Contradiction found")
                break
            if(self.verbose):
                print("Iterated logic.")
                print("Puzzle size: %d"%self.naive_size())
                print("Solved elements: %d"%self.num_solved())

if __name__ == "__main__":
    # empty = Sudoku([
    #     [None, None, None, None, None, None, None, None, None],
    #     [None, None, None, None, None, None, None, None, None],
    #     [None, None, None, None, None, None, None, None, None],
    #     [None, None, None, None, None, None, None, None, None],
    #     [None, None, None, None, None, None, None, None, None],
    #     [None, None, None, None, None, None, None, None, None],
    #     [None, None, None, None, None, None, None, None, None],
    #     [None, None, None, None, None, None, None, None, None],
    #     [None, None, None, None, None, None, None, None, None]
    # ])
    # game1 = Sudoku([
    #     [5, 3, None, None, 7, None, None, None, None],
    #     [6, None, None, 1, 9, 5, None, None, None],
    #     [None, 9, 8, None, None, None, None, 6, None],
    #     [8, None, None, None, 6, None, None, None, 3],
    #     [4, None, None, 8, None, 3, None, None, 1],
    #     [7, None, None, None, 2, None, None, None, 6],
    #     [None, 6, None, None, None, None, 2, 8, None],
    #     [None, None, None, 4, 1, 9, None, None, 5],
    #     [None, None, None, None, 8, None, None, 7, 9]
    # ], verbose=True)
    
    # game1.solve()
    
    # print(game1)
    
    evil1 = Sudoku([
        [None, None, None, None, None, None, None, None, 4],
        [None, None, 9, None, 8, None, 2, None, 1],
        [None, 3, None, None, None, 9, None, None, None],
        [None, 5, None, 1, None, None, 6, None, 2],
        [None, None, None, None, 6, None, None, 3, None],
        [None, None, 2, None, None, None, None, 4, None],
        [None, None, None, None, None, None, None, 6, None],
        [7, None, None, 5, None, None, None, None, None],
        [None, None, 3, None, 1, None, 8, None, 9]
    ], verbose=True)
    
    evil1.solve()
    
    print(evil1)