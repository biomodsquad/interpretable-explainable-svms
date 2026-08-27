import numpy as np
import random

class cvSet():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.train = []
        self.test = []
        self.type = None

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def classification(self, num_sets = 5, validation_size = 0.2, random_seed = 0):
        self.type = "classification"
        random.seed(random_seed)
        
        classes = np.unique(self.y)
        class_count = []
        class_ind = []
        for c in classes:
            is_class = (self.y == c)
            class_count.append(sum(is_class+0))
            
            ind = [i for i, val in enumerate(is_class) if val]
            class_ind.append(ind)
            
        num_class_val = (np.array(class_count)*validation_size).astype(int)

        for s in range(num_sets):
            val_set = []
            for c in range(len(classes)):
                val_set += random.sample(class_ind[c], num_class_val[c])
            
            train_set = [i for i in range(len(self.y))]
            for v in val_set:
                train_set.remove(v)
            
            self.train.append(np.array(train_set))
            self.test.append(np.array(val_set))

    def k_fold(self, num_folds = 5):
        self.type = "k-fold"
        
        all_ind = list(range(len(self.y)))
        
        self.sets = []
        for f in range(num_folds):
            test_ind = list(range(f,len(self.y),num_folds))
            train_ind = list(set(all_ind).difference(test_ind))

            self.train.append(np.array(train_ind))
            self.test.append(np.array(test_ind))

    def independent(self, num_sets = 5, validation_size = 0.2, random_seed = 0):
        self.type = "independent"
        random.seed(random_seed)
        
        classes = np.unique(self.y)
        class_count = []
        class_ind = []
        for c in classes:
            is_class = (self.y == c)
            class_count.append(sum(is_class+0))
            
            ind = [i for i, val in enumerate(is_class) if val]
            class_ind.append(ind)
            
        num_class = np.round(np.array(class_count)/num_sets).astype(int)

        for s in range(num_sets-1):
            train_set = []
            val_set = []
            for c in range(len(classes)):
                selected_class_ind = random.sample(class_ind[c], num_class[c])
                for ind in selected_class_ind:
                    class_ind[c].remove(ind)
                    
                selected_val_ind = random.sample(selected_class_ind, 
                                                 np.round(num_class[c]*validation_size).astype(int))
                for v in selected_val_ind:
                    selected_class_ind.remove(v)

                val_set += selected_val_ind
                train_set += selected_class_ind
            
            self.train.append(np.array(train_set))
            self.test.append(np.array(val_set))

        train_set = []
        val_set = []
        for c in range(len(classes)):
            selected_class_ind = class_ind[c]
                
            selected_val_ind = random.sample(selected_class_ind, 
                                             np.round(num_class[c]*validation_size).astype(int))
            for v in selected_val_ind:
                selected_class_ind.remove(v)

            val_set += selected_val_ind
            train_set += selected_class_ind
        
        self.train.append(np.array(train_set))
        self.test.append(np.array(val_set))
        
