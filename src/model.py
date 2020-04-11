import numpy as np
from abc import ABC, abstractmethod
import hyperplane

class RegionMemory():
    memory = {}
    def __init_(self):
        pass

    @staticmethod
    def hash(z_left, z_right):
        return '|'.join(z_left.astype(str)) + ':' + '|'.join(z_right.astype(str))

    # Memory add
    def add(self, z_left, z_right, value):
        lookup = self.__class__.hash(z_left, z_right)
        self.memory[lookup] = value

    # Memory lookup
    def lookup(self, z_left, z_right):
        lookup = self.__class__.hash(z_left, z_right)
        if lookup in self.memory:
            return self.memory[lookup]
        else:
            return None

class BAT(ABC):
    dp_memory = RegionMemory()
    lookup_memory = RegionMemory()
    model = None
    c_hyperplanes = None
    epsilon = None
    log = False

    @staticmethod
    def validate(z):
        assert isinstance(z, np.ndarray)
        assert len(z.squeeze().shape) == 1
        return z

    # Evaluate a region
    def evaluate_region(self, z):
        c_mem = self.lookup_memory.lookup(z, z)
        if c_mem: 
            return c_mem
        c_point = np.zeros_like(z, dtype = float)
        it = np.nditer(z, flags=['c_index'])
        while not it.finished:
            if (it[0] == 1):
                c_point[it.index] = self.c_hyperplanes[it.index][0] - self.epsilon
            elif (it[0] > len(self.c_hyperplanes[it.index])):
                c_point[it.index] = self.c_hyperplanes[it.index][-1] + self.epsilon
            else:
                c_point[it.index] = np.mean((self.c_hyperplanes[it.index][it[0] - 1], self.c_hyperplanes[it.index][it[0] - 2]))
            it.iternext()
        prediction = self.model.predict(c_point.reshape(1, -1)).squeeze()
        self.lookup_memory.add(z, z, prediction)
        return prediction

    # Optimize function 
    @abstractmethod
    def born_again(self, z_left, z_right):
        pass

    def fit(self):
        print(self.born_again(np.ones(len(self.c_hyperplanes), dtype = int), 
                              np.array([len(i) for i in self.c_hyperplanes], dtype = int)))
        return self

    # Initialise class
    def __init__(self, model_in, epsilon = 0.001, log = False):
        self.model = model_in
        self.c_hyperplanes = hyperplane.extract_hyperplanes(model_in)
        self.epsilon = epsilon
        self.log = log


class BATDepth(BAT):

    def __init__(self, model, epsilon = 0.001, log = False):
        super().__init__(model, epsilon, log)

    def born_again(self, z_left, z_right):
        z_left = super(BATDepth, self).validate(z_left)
        z_right = super(BATDepth, self).validate(z_right)

        # In the same region, so return 0 depth
        if (z_left == z_right).all(): 
            return 0
        
        # Check memory for object
        c_mem = self.dp_memory.lookup(z_left, z_right)
        if c_mem: return c_mem
        
        if self.log:
            print('Call - Z_left: {}, Z_right: {}'.format(z_left, z_right))

        # The depth of the full region is always larger than that of any subregion
        upper_bound = np.inf
        lower_bound = 0

        for j in range(len(self.c_hyperplanes)):
            if not lower_bound < upper_bound: break

            # Set lower and upper bound for feature
            low, up = z_left[j], z_right[j]

            while low < up and lower_bound < upper_bound:
                # Binary search - pick middle point in region
                l_c = int((low+up)/2)
                one_hot_j = np.zeros_like(z_left)
                one_hot_j[j] = 1

                # Calculate depth on LHS and RHS
                depth_l_r_sub = self.born_again(z_left, z_right + one_hot_j * (l_c - z_right[j]))
                depth_l_sub_r = self.born_again(z_left + one_hot_j*(l_c + 1 - z_left[j]), z_right)

                if ((depth_l_r_sub == 0) and (depth_l_sub_r == 0)):
                    if (self.evaluate_region(z_left) == self.evaluate_region(z_right)):
                        self.dp_memory.add(z_left, z_right, 0)
                        if self.log: 
                            print('Special-case evaluated: 0')
                        return 0
                    else:
                        self.dp_memory.add(z_left, z_right, 1)
                        if self.log:
                            print('Special-case evaluated: 1')
                        return 1

                if self.log:
                    print('UB: {}, LB: {}, up: {}, low: {}, phi_1: {}, phi_2: {}'.format(upper_bound, lower_bound, up, low, depth_l_r_sub, depth_l_sub_r))
                    
                upper_bound = min(upper_bound, 1 + max(depth_l_r_sub, depth_l_sub_r))
                lower_bound = max(lower_bound, max(depth_l_r_sub, depth_l_sub_r))
                
                if depth_l_r_sub >= depth_l_sub_r:
                    up = l_c
                if depth_l_r_sub <= depth_l_sub_r:
                    low = l_c + 1
                        
        self.dp_memory.add(z_left, z_right, upper_bound)
        return upper_bound