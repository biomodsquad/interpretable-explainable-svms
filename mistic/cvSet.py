"""Cross-validation split generation for MISTIC models."""

import numpy as np
import random

class cvSet():
    """Store a dataset and generate reusable train/test index sets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target values.

    Attributes
    ----------
    train, test : list of numpy.ndarray
        Training and test indices for each generated split.
    type : {"classification", "regression", None}
        Split strategy most recently used.
    """

    def __init__(self, X, y):
        """Initialize the split container from features and targets."""
        self.X = X
        self.y = y
        self.train = []
        self.test = []
        self.type = None

    def __getstate__(self):
        """Return instance state for pickle serialization."""
        return self.__dict__

    def __setstate__(self, state):
        """Restore instance state from a pickle payload."""
        self.__dict__.update(state)

    def classification(self, num_sets = 5, validation_size = 0.2, random_seed = 0):
        """Generate stratified random splits for binary classification.

        Parameters
        ----------
        num_sets : int, default=5
            Number of train/test splits to generate.
        validation_size : float, default=0.2
            Fraction of each class assigned to the test set.
        random_seed : int, default=0
            Seed used by the split generator.
        """
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
        """Generate shuffled folds for regression.

        Parameters
        ----------
        num_folds : int, default=5
            Number of folds. Each fold is used once as the test set.
        """
        self.type = "k-fold"
        
        all_ind = list(range(len(self.y)))
        
        self.sets = []
        for f in range(num_folds):
            test_ind = list(range(f,len(self.y),num_folds))
            train_ind = list(set(all_ind).difference(test_ind))

            self.train.append(np.array(train_ind))
            self.test.append(np.array(test_ind))

    def independent(self, num_sets = 5, validation_size = 0.2, random_seed = 0):
        """Generate independent random holdout splits for regression.

        Parameters
        ----------
        num_sets : int, default=5
            Number of train/test splits to generate.
        validation_size : float, default=0.2
            Fraction of samples assigned to each test set.
        random_seed : int, default=0
            Seed used by the split generator.
        """
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
        
