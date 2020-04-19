import numpy as np
from abc import ABC, abstractmethod
import hyperplane
import tree

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
        if (z_left == z_right).all():
            return 0
        lookup = self.__class__.hash(z_left, z_right)
        if lookup in self.memory:
            return self.memory[lookup]
        else:
            return None


class BAT(ABC):
    dp_memory = RegionMemory()
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
        c_point = np.zeros_like(z, dtype=float)
        it = np.nditer(z, flags=['c_index'])
        while not it.finished:
            if (it[0] == 1):
                if len(self.c_hyperplanes[it.index]) == 0:
                    # Does not matter (RF does not split on feature)
                    c_point[it.index] = 0
                else:
                    c_point[it.index] = self.c_hyperplanes[it.index][0] - \
                        self.epsilon  # Epsilon before first split
            elif (it[0] == len(self.c_hyperplanes[it.index]) + 1):
                c_point[it.index] = self.c_hyperplanes[it.index][-1] + \
                    self.epsilon  # Epsilon after last split
            else:
                # Region z_ji defines a point between split j_(i-2) and j_(i-1)
                c_point[it.index] = np.mean(
                    (self.c_hyperplanes[it.index][it[0] - 1], self.c_hyperplanes[it.index][it[0] - 2]))
            it.iternext()
        prediction = self.model.predict(c_point.reshape(1, -1)).squeeze()
        return prediction

    # Optimize function
    @abstractmethod
    def born_again(self, z_left, z_right):
        pass
    
    @abstractmethod
    def calculate_objective(self, phi_one, phi_two):
        pass

    # Re-create decision tree function
    def extract_optimal_solutions(self, z_left, z_right, phi_opt):
        if phi_opt == 0:
            return ['leaf', self.evaluate_region(z_left)]
        else:
            if self.log:
                print('Extracting optimal solution for: ', z_left, z_right, phi_opt)
            out_list = []
            for j in range(len(self.c_hyperplanes)):
                for l_c in range(z_left[j], z_right[j]):
                    one_hot_j = np.zeros_like(z_left, dtype = int)
                    one_hot_j[j] = 1
                    z_right_mid = z_right + one_hot_j * (l_c - z_right[j])
                    phi_one = self.dp_memory.lookup(z_left, z_right_mid)
                    if phi_one is None: 
                        continue
                    z_left_mid = z_left + one_hot_j * (l_c + 1 - z_left[j])
                    phi_two = self.dp_memory.lookup(z_left_mid, z_right)
                    if phi_two is None: 
                        continue
                    if self.log:
                        print('Evaluating split: ', phi_opt, '|' , z_left, phi_one, z_right_mid, ':', z_left_mid, phi_two, z_right_mid)
                    if phi_opt == self.calculate_objective(phi_one, phi_two):
                        if self.log:
                            print('Valid split!')
                        threshold = np.inf
                        if l_c <= len(self.c_hyperplanes[j]):
                            threshold = self.c_hyperplanes[j][l_c - 1]
                        return {
                            'node': ['split', j, threshold], 
                            'left_child': self.extract_optimal_solutions(z_left, z_right_mid, phi_one),
                            'right_child': self.extract_optimal_solutions(z_left_mid, z_right, phi_two)
                        }

    def fit(self):
        z_bound_left, z_bound_right = hyperplane.get_bounds(self.c_hyperplanes)
        phi_opt = self.born_again(z_bound_left, z_bound_right)
        return tree.DecisionTree(self.extract_optimal_solutions(z_bound_left, z_bound_right, phi_opt), self.columns)

    # Initialise class
    def __init__(self, model_in, columns, log, epsilon=0.001):
        self.model = model_in
        self.columns = columns
        self.c_hyperplanes = hyperplane.extract_hyperplanes(model_in)
        self.epsilon = epsilon
        self.log = log


class BATDepth(BAT):

    def __init__(self, model, epsilon=0.001, log=False):
        super().__init__(model, epsilon, log)

    def calculate_objective(self, phi_one, phi_two):
        return max([phi_one, phi_two]) + 1

    def born_again(self, z_left, z_right):
        z_left = super(BATDepth, self).validate(z_left)
        z_right = super(BATDepth, self).validate(z_right)

        # In the same region, so return 0 depth
        if (z_left == z_right).all():
            return 0

        # Check memory for object
        c_mem = self.dp_memory.lookup(z_left, z_right)
        if c_mem:
            return c_mem

        # The depth of the full region is always larger than that of any subregion
        upper_bound = np.inf
        lower_bound = 0

        for j in range(len(self.c_hyperplanes)):
            if not lower_bound < upper_bound:
                break

            # Set lower and upper bound for feature
            low, up = z_left[j], z_right[j]

            while low < up and lower_bound < upper_bound:
                # Binary search - pick middle point in region
                l_c = int(np.floor((low+up)/2))
                one_hot_j = np.zeros_like(z_left, dtype = int)
                one_hot_j[j] = 1

                z_right_mid = z_right + one_hot_j * (l_c - z_right[j])
                z_left_mid = z_left + one_hot_j*(l_c + 1 - z_left[j])
                if self.log:
                    print(
                        'Calling:'
                        'Z_left', z_left,
                        'Z_L_R_sub', z_right_mid,
                        'Z_L_sub_R', z_left_mid,
                        'Z_right', z_right
                    )

                # Calculate depth on LHS and RHS
                depth_l_r_sub = self.born_again(
                    z_left, z_right_mid)
                depth_l_sub_r = self.born_again(
                    z_left_mid, z_right)

                if ((depth_l_r_sub == 0) and (depth_l_sub_r == 0)):
                    if (self.evaluate_region(z_left) == self.evaluate_region(z_right)):
                        self.dp_memory.add(z_left, z_right, 0)
                        if self.log:
                            print('Special-case evaluated: Contiguous', z_left, z_right)
                        return 0
                    else:
                        self.dp_memory.add(z_left, z_right, 1)
                        if self.log:
                            print('Special-case evaluated: Non-Contiguous', z_left, z_right)
                        return 1

                if self.log:
                    print('UB: {}, LB: {}, up: {}, low: {}, phi_1: {}, phi_2: {}'.format(
                        upper_bound, lower_bound, up, low, depth_l_r_sub, depth_l_sub_r))

                upper_bound = np.min(
                    [upper_bound, 
                    1 + np.max([depth_l_r_sub, depth_l_sub_r])])
                lower_bound = np.max(
                    [lower_bound, 
                    np.max([depth_l_r_sub, depth_l_sub_r])])

                if depth_l_r_sub >= depth_l_sub_r:
                    up = l_c
                if depth_l_r_sub <= depth_l_sub_r:
                    low = l_c + 1

        self.dp_memory.add(z_left, z_right, upper_bound)
        return upper_bound