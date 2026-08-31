import numpy as np
import random

class cvSet():
    def __init__(self, X, y, num_feature_medoids=20,
                 ensemble_validation_size=0.0,
                 ensemble_validation_random_seed=0,
                 ensemble_validation_stratify=False):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        if len(self.X) != len(self.y):
            raise ValueError("X and y must contain the same number of samples")
        self.train = []
        self.test = []
        self.type = None
        self.ensemble_validation_size = self._validate_ensemble_validation_size(
            ensemble_validation_size)
        self.ensemble_validation_random_seed = ensemble_validation_random_seed
        if not isinstance(ensemble_validation_stratify, (bool, np.bool_)):
            raise TypeError("ensemble_validation_stratify must be boolean")
        self.ensemble_validation_stratify = bool(
            ensemble_validation_stratify)
        self._initialize_ensemble_validation_set()
        self.num_feature_medoids = self._validate_num_feature_medoids(
            num_feature_medoids)
        self.feature_medoids_ = self._feature_medoids(
            self.X[self.development_indices_], self.num_feature_medoids)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "num_feature_medoids" not in self.__dict__:
            self.num_feature_medoids = min(20, self.X.shape[1])
        if "feature_medoids_" not in self.__dict__:
            self.feature_medoids_ = self._feature_medoids(
                self.X, self.num_feature_medoids)
        if "ensemble_validation_indices_" not in self.__dict__:
            self.ensemble_validation_size = 0.0
            self.ensemble_validation_random_seed = 0
            self.ensemble_validation_indices_ = np.array([], dtype=int)
            self.development_indices_ = np.arange(len(self.y), dtype=int)
        if "ensemble_validation_stratify" not in self.__dict__:
            self.ensemble_validation_stratify = False

    def _validate_ensemble_validation_size(self, validation_size):
        if (isinstance(validation_size, (bool, np.bool_)) or
                not isinstance(validation_size, (int, float, np.integer,
                                                  np.floating))):
            raise TypeError("ensemble_validation_size must be numeric")
        validation_size = float(validation_size)
        if not 0 <= validation_size < 1:
            raise ValueError("ensemble_validation_size must be in [0, 1)")
        if validation_size > 0 and len(self.y) < 2:
            raise ValueError(
                "an ensemble validation set requires at least two samples")
        return validation_size

    def _initialize_ensemble_validation_set(self):
        """Reserve samples that no feature-selection CV split may use."""
        count = int(np.floor(len(self.y) * self.ensemble_validation_size))
        if self.ensemble_validation_size > 0:
            count = max(1, count)
        count = min(count, max(0, len(self.y) - 1))
        rng = np.random.default_rng(self.ensemble_validation_random_seed)
        if self.ensemble_validation_stratify and count:
            classes, class_counts = np.unique(self.y, return_counts=True)
            exact_counts = count * class_counts / len(self.y)
            validation_counts = np.floor(exact_counts).astype(int)
            remaining = count - np.sum(validation_counts)
            remainder_order = np.argsort(
                -(exact_counts - validation_counts), kind="stable")
            for class_index in remainder_order[:remaining]:
                validation_counts[class_index] += 1
            selected = []
            for class_value, class_count in zip(classes, validation_counts):
                class_indices = np.flatnonzero(self.y == class_value)
                selected.extend(rng.choice(
                    class_indices, size=class_count, replace=False).tolist())
            self.ensemble_validation_indices_ = np.sort(
                np.asarray(selected, dtype=int))
        else:
            self.ensemble_validation_indices_ = np.sort(
                rng.choice(len(self.y), size=count, replace=False)).astype(int)
        development_mask = np.ones(len(self.y), dtype=bool)
        development_mask[self.ensemble_validation_indices_] = False
        self.development_indices_ = np.flatnonzero(development_mask)

    def _validate_num_feature_medoids(self, num_feature_medoids):
        if not isinstance(num_feature_medoids, (int, np.integer)):
            raise TypeError("num_feature_medoids must be an integer")
        if num_feature_medoids < 1:
            raise ValueError("num_feature_medoids must be at least 1")
        return min(int(num_feature_medoids), self.X.shape[1])

    @staticmethod
    def _feature_medoids(X, num_medoids, max_iter=100):
        """Cluster normalized feature profiles and return medoid indices."""
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature")
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError("K-medoids preprocessing requires numeric features")

        profiles = np.asarray(X, dtype=float).T
        profiles = profiles - np.mean(profiles, axis=1, keepdims=True)
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        profiles = np.divide(
            profiles, norms, out=np.zeros_like(profiles), where=norms != 0)

        squared_norms = np.sum(profiles * profiles, axis=1)
        distances = (squared_norms[:, None] + squared_norms[None, :]
                     - 2 * np.matmul(profiles, profiles.T))
        distances = np.maximum(distances, 0)

        # Deterministic farthest-first initialization, beginning with the
        # feature having the smallest total distance to all other features.
        medoids = [int(np.argmin(np.sum(distances, axis=1)))]
        nearest_distance = distances[:, medoids[0]].copy()
        while len(medoids) < num_medoids:
            candidate_distance = nearest_distance.copy()
            candidate_distance[medoids] = -np.inf
            next_medoid = int(np.argmax(candidate_distance))
            medoids.append(next_medoid)
            nearest_distance = np.minimum(
                nearest_distance, distances[:, next_medoid])

        medoids = np.asarray(medoids, dtype=int)
        for _ in range(max_iter):
            labels = np.argmin(distances[:, medoids], axis=1)
            updated = medoids.copy()
            for cluster_index in range(num_medoids):
                members = np.flatnonzero(labels == cluster_index)
                if members.size:
                    within_cluster = distances[np.ix_(members, members)]
                    updated[cluster_index] = members[
                        np.argmin(np.sum(within_cluster, axis=1))]
            if np.array_equal(updated, medoids):
                break
            medoids = updated

        return np.sort(medoids)

    def _reset_splits(self):
        """Discard previously generated splits before configuring new ones."""
        self.train = []
        self.test = []

    def classification(self, num_sets = 5, validation_size = 0.2, random_seed = 0):
        self._reset_splits()
        self.type = "classification"
        random.seed(random_seed)
        
        development_y = self.y[self.development_indices_]
        classes = np.unique(development_y)
        class_count = []
        class_ind = []
        for c in classes:
            is_class = (development_y == c)
            class_count.append(np.count_nonzero(is_class))
            
            ind = self.development_indices_[is_class].tolist()
            class_ind.append(ind)
            
        num_class_val = (np.array(class_count)*validation_size).astype(int)

        for s in range(num_sets):
            val_set = []
            for c in range(len(classes)):
                val_set += random.sample(class_ind[c], num_class_val[c])
            
            is_training = np.zeros(len(self.y), dtype=bool)
            is_training[self.development_indices_] = True
            is_training[val_set] = False
            train_set = np.flatnonzero(is_training)
            
            self.train.append(train_set)
            self.test.append(np.array(val_set))

    def k_fold(self, num_folds = 5):
        self._reset_splits()
        self.type = "k-fold"
        
        self.sets = []
        for f in range(num_folds):
            test_ind = self.development_indices_[f::num_folds]
            is_training = np.zeros(len(self.y), dtype=bool)
            is_training[self.development_indices_] = True
            is_training[test_ind] = False
            train_ind = np.flatnonzero(is_training)

            self.train.append(train_ind)
            self.test.append(test_ind)

    def independent(self, num_sets = 5, validation_size = 0.2, random_seed = 0):
        self._reset_splits()
        self.type = "independent"
        random.seed(random_seed)
        
        development_y = self.y[self.development_indices_]
        classes = np.unique(development_y)
        class_count = []
        class_ind = []
        for c in classes:
            is_class = (development_y == c)
            class_count.append(np.count_nonzero(is_class))
            
            ind = self.development_indices_[is_class].tolist()
            class_ind.append(ind)
            
        num_class = np.round(np.array(class_count)/num_sets).astype(int)

        for s in range(num_sets-1):
            train_set = []
            val_set = []
            for c in range(len(classes)):
                selected_class_ind = random.sample(class_ind[c], num_class[c])
                selected_class_set = set(selected_class_ind)
                class_ind[c] = [
                    ind for ind in class_ind[c]
                    if ind not in selected_class_set]
                    
                selected_val_ind = random.sample(selected_class_ind, 
                                                 np.round(num_class[c]*validation_size).astype(int))
                selected_val_set = set(selected_val_ind)

                val_set += selected_val_ind
                train_set += [
                    ind for ind in selected_class_ind
                    if ind not in selected_val_set]
            
            self.train.append(np.array(train_set))
            self.test.append(np.array(val_set))

        train_set = []
        val_set = []
        for c in range(len(classes)):
            selected_class_ind = class_ind[c]
                
            selected_val_ind = random.sample(selected_class_ind, 
                                             np.round(num_class[c]*validation_size).astype(int))
            selected_val_set = set(selected_val_ind)

            val_set += selected_val_ind
            train_set += [
                ind for ind in selected_class_ind
                if ind not in selected_val_set]
        
        self.train.append(np.array(train_set))
        self.test.append(np.array(val_set))
