import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   

from scipy.optimize import minimize

import copy
from collections import Counter

from sklearn.svm import SVR

import random

from mistic.utility import combined_rank, kernelWrapper, score_svr, score_svc, dotdict, svc_dec2, rank_items

class svmSet():
        
    def __init__(self, SVM, cvSet, score_method, 
                 kernel = None,
                 separate_feature_sets = False,
                 separate_parameters = False,
                 perturbation_sets = None):
        self.SVM = SVM
        self.cv = cvSet
        
        self.num_samples = self.cv.X.shape[0]
        self.num_models = len(self.cv.train)
        self.separate_feature_sets = separate_feature_sets
        self.separate_parameters = separate_parameters
        
        if self.separate_feature_sets:
            self.features = []
            self.removed_features_ = []
            for i in range(self.num_models):
                self.features.append(np.array([f for f in range(self.cv.X.shape[1])]))
                self.removed_features_.append([])
        else:
            self.features = np.array([f for f in range(self.cv.X.shape[1])])
            self.removed_features_ = []

        all_features = np.arange(self.cv.X.shape[1])
        if perturbation_sets is None:
            self.perturbation_sets = [[int(feature)] for feature in all_features]
        else:
            self.perturbation_sets = self._normalize_perturbation_sets(
                perturbation_sets, all_features)

        self.kernel = kernelWrapper() if kernel is None else kernel
        self._reset_kernel_matrix()
        self._kernel_configuration_ = None
    
        self.score = score_method

        self.models = []
        self.X_ind = []
        for i in range(self.num_models):
            self.models.append(copy.deepcopy(self.SVM))
            self.X_ind.append(self.cv.train[i])
    
    
    def __getstate__(self):
            return self.__dict__

    def __setstate__(self, state):
            self.__dict__.update(state)
            # Older serialized objects predate kernel-configuration tracking.
            if "_kernel_configuration_" not in self.__dict__:
                self._kernel_configuration_ = None
            if "perturbation_sets" not in self.__dict__:
                self.perturbation_sets = [
                    [feature] for feature in range(self.cv.X.shape[1])]
    
    def _train_models(self):
        for i in range(self.num_models):
            if self.separate_feature_sets | self.separate_parameters:
                kernel_matrix = self._get_kernel_matrix(self.X_ind[i],self.X_ind[i],model_index = i)
            else:
                kernel_matrix = self._get_kernel_matrix(self.X_ind[i],self.X_ind[i])
                
            self.models[i].fit(kernel_matrix, self.cv.y[self.X_ind[i]])

    
    def _update_kernel_matrix(self):
        if self.separate_feature_sets | self.separate_parameters:
            for i in range(self.num_models):     
                if self.separate_feature_sets:
                    features = self.features[i]
                else:
                    features = self.features
                    
                if isinstance(self.parameters_,list):
                    parameters = self.parameters_[i].kernel
                else:
                    parameters = self.parameters_.kernel
                    
                # A fold only ever needs columns belonging to its training
                # set. Avoid retaining a full square matrix for every model.
                self.kernel_matrix_[i] = self.kernel.compute(
                    self.cv.X,
                    feature_index = features,
                    parameters = parameters,
                    Y = self.cv.X[self.cv.train[i], :])
        else:
            self.kernel_matrix_ = self.kernel.compute(self.cv.X, 
                                                      feature_index = self.features, 
                                                      parameters = self.parameters_.kernel)

    
    def _reset_kernel_matrix(self):
        if self.separate_feature_sets | self.separate_parameters:
            self.kernel_matrix_ = []
            for i in range(self.num_models):
                self.kernel_matrix_.append(
                    np.zeros((self.num_samples, len(self.cv.train[i]))))
        else:
            self.kernel_matrix_ = np.zeros((self.num_samples, self.num_samples))

    
    def _get_kernel_matrix(self,indices_1, indices_2, model_index = None):
        if isinstance(self.kernel_matrix_,list):
            # Per-model matrices contain training columns only. Translate
            # sample indices to their compact column positions.
            train_indices = self.cv.train[model_index]
            positions = {sample: position for position, sample in enumerate(train_indices)}
            column_indices = np.fromiter(
                (positions[sample] for sample in indices_2),
                dtype=int,
                count=len(indices_2))
            kernel_matrix = self.kernel_matrix_[model_index][np.ix_(indices_1, column_indices)]
        else:
            kernel_matrix = self.kernel_matrix_[indices_1, :][:, indices_2]

        return kernel_matrix

    
    def _score_models(self):  
        accuracy = []
        for i in range(self.num_models):
            score = self.score(self,model_index = i)
            
            if self.separate_parameters:
                accuracy.append(score)
            else:    
                if i == 0:
                    accuracy = score
                else:
                    accuracy = {key: accuracy[key]+score[key] for key in score.keys()}
            
        if self.separate_parameters:
            self.performance_ = accuracy
            
            for i in range(self.num_models):
                if isinstance(self.parameters_,list):
                    parameters = self.parameters_[i]
                else:
                    parameters = self.parameters_

                self.performance_[i] = dotdict(self.performance_[i])
                self.performance_[i].update(parameters.model) 
                self.performance_[i].update(parameters.kernel)
        else:    
            self.performance_ = dotdict({key: accuracy[key]/self.num_models for key in accuracy.keys()})
            self.performance_.update(self.parameters_.model) 
            self.performance_.update(self.parameters_.kernel)

    
    def tune_models(self, parameter_grid):
        if self.separate_parameters:
            best_score = self.num_models*[-1e12]
            best_models = self.num_models*[0]
            best_kernel_matrix = self.num_models*[0]
            best_parameters = self.num_models*[0]
            best_performance = self.num_models*[0]
        else:
            best_score = -1e12
            
        tune_performance = {}
        result = 0
        for parameter_set in parameter_grid:
            self._update_parameters(parameter_set)
            self._train_models()
            self._score_models()
            tune_performance[result] = self.performance_
            result += 1

            if self.separate_parameters:
                for i in range(self.num_models):
                    if self.performance_[i].score > best_score[i]:
                        best_models[i] = copy.deepcopy(self.models[i])
                        best_kernel_matrix[i] = copy.deepcopy(self.kernel_matrix_[i])
                        if isinstance(parameter_set, list):
                            best_parameters[i] = copy.deepcopy(parameter_set[i])
                        else:
                            # A single candidate is broadcast to every fold;
                            # each fold can still select it independently.
                            best_parameters[i] = copy.deepcopy(parameter_set)
                        best_performance[i] = copy.deepcopy(self.performance_[i])
                        best_score[i] = self.performance_[i].score
            else:
                if self.performance_.score > best_score:
                    best_models = copy.deepcopy(self.models)
                    best_kernel_matrix = copy.deepcopy(self.kernel_matrix_)
                    best_parameters = parameter_set
                    best_performance = self.performance_
                    best_score = self.performance_.score
    
        self.tune_performance_ = tune_performance

        self.models = best_models
        self.performance_ = best_performance
        
        # The winning matrix was retained above, so only restore model and
        # parameter attributes here; rebuilding its kernel would be wasted.
        self._update_parameters(best_parameters, update_kernel = False)
        self.kernel_matrix_ = best_kernel_matrix
        self._kernel_configuration_ = self._kernel_configuration(best_parameters)

    
    def _reduce_models(self,parameter_grid):
        for i in range(self.num_models):
            self.X_ind[i] = self.X_ind[i][self.models[i].support_]
            
        self.tune_models(parameter_grid)


    def _reset_X_ind(self,parameter_grid):
        for i in range(self.num_models):
            self.X_ind[i] = self.cv.train[i]

        self.tune_models(parameter_grid)

    
    def _get_support_vectors(self,model_index):
        return self.cv.X[self.X_ind[model_index],:][self.models[model_index].support_,:]
        

    @staticmethod
    def _freeze_parameter(value):
        """Convert nested parameter values into a comparable signature."""
        if isinstance(value, dict):
            return tuple(sorted(
                (key, svmSet._freeze_parameter(item))
                for key, item in value.items()))
        if isinstance(value, np.ndarray):
            return (value.dtype.str, value.shape, value.tobytes())
        if isinstance(value, (list, tuple)):
            return tuple(svmSet._freeze_parameter(item) for item in value)
        try:
            hash(value)
            return value
        except TypeError:
            return repr(value)

    def _kernel_configuration(self, parameter_set):
        if isinstance(parameter_set, list):
            kernel_parameters = [parameters.kernel for parameters in parameter_set]
        else:
            kernel_parameters = parameter_set.kernel

        return (
            self._freeze_parameter(kernel_parameters),
            self._freeze_parameter(self.features))


    def _update_parameters(self, parameter_set, update_kernel = True):
        self.parameters_ = parameter_set
        
        if isinstance(parameter_set,list):
            for i in range(self.num_models):
                for model_param in parameter_set[i].model.keys():
                        setattr(self.models[i],model_param, parameter_set[i].model[model_param])
        else:
            for i in range(self.num_models):
                for model_param in parameter_set.model.keys():
                        setattr(self.models[i],model_param, parameter_set.model[model_param])

        kernel_configuration = self._kernel_configuration(parameter_set)
        if update_kernel and kernel_configuration != self._kernel_configuration_:
            self._reset_kernel_matrix()
            self._update_kernel_matrix()
            self._kernel_configuration_ = kernel_configuration

    
    def _remove_features(self, to_remove, model_index = None,
                         update_kernel = True):
        self._reset_kernel_matrix()
        self._kernel_configuration_ = None

        if model_index is not None:
            current_features = self.features[model_index]
        else:
            current_features = self.features

        # Features are selection units only through their perturbation set.
        # Selecting any member removes every active member of that set.
        requested = set(np.asarray(to_remove).ravel().tolist())
        expanded_removal = []
        for perturbation_set in self.perturbation_sets:
            if requested.intersection(perturbation_set):
                expanded_removal.extend(perturbation_set)
        expanded_removal = np.asarray([
            feature for feature in expanded_removal if feature in current_features])
        current_features = current_features[
            ~np.isin(current_features, expanded_removal)]
        
        if model_index is not None:
            self.features[model_index] = current_features
            self.removed_features_[model_index] = np.append(
                self.removed_features_[model_index], expanded_removal)
        else:
            self.features = current_features
            self.removed_features_ = np.append(
                self.removed_features_, expanded_removal)
            
        if len(current_features) > 0 and update_kernel:
            self._update_kernel_matrix()
            self._kernel_configuration_ = self._kernel_configuration(self.parameters_)


    def _set_features(self, features, model_index=None, update_kernel=True):
        """Replace the active feature set while preserving its original order."""
        requested = set(np.asarray(features).ravel().tolist())
        ordered = np.asarray([
            feature for group in self.perturbation_sets for feature in group
            if feature in requested], dtype=int)

        self._reset_kernel_matrix()
        self._kernel_configuration_ = None
        if model_index is None:
            self.features = ordered
        else:
            self.features[model_index] = ordered

        if len(ordered) > 0 and update_kernel:
            self._update_kernel_matrix()
            self._kernel_configuration_ = self._kernel_configuration(self.parameters_)


    def _add_features(self, to_add, model_index=None, update_kernel=True):
        """Add complete perturbation sets to the active feature set."""
        current = (self.features if model_index is None
                   else self.features[model_index])
        requested = set(np.asarray(to_add).ravel().tolist())
        expanded = list(current)
        for perturbation_set in self.perturbation_sets:
            if requested.intersection(perturbation_set):
                expanded.extend(perturbation_set)
        self._set_features(expanded, model_index, update_kernel)
            
    
    def greedy_backward_selection(self, parameter_grid, 
                                  reduction_factor = 0.1, 
                                  feature_ranker = combined_rank().compute, 
                                  set_for_rank = "train",
                                  tune_models_each_step = True):
        """Rank and remove feature sets using greedy backward selection.

        When ``tune_models_each_step`` is false, tuning is performed only for
        the initial full-feature model. Its selected parameters are retained,
        while gamma, when present, is scaled at each later step as
        ``initial_gamma * initial_feature_count / current_feature_count``.
        Kernels without a gamma parameter retain their tuned parameters.
        """
        
        feature_performance = {}
        result = 0
        best_score = -1e12
        initial_feature_counts = np.asarray([
            len(self.features[index]) if self.separate_feature_sets
            else len(self.features)
            for index in range(self.num_models)])
        baseline_parameters = None

        def fit_current_feature_set():
            nonlocal baseline_parameters
            if tune_models_each_step or baseline_parameters is None:
                self.tune_models(parameter_grid)
                if baseline_parameters is None:
                    baseline_parameters = copy.deepcopy(self.parameters_)
                return

            scaled_parameters = copy.deepcopy(baseline_parameters)
            current_feature_counts = np.asarray([
                len(self.features[index]) if self.separate_feature_sets
                else len(self.features)
                for index in range(self.num_models)])

            if isinstance(scaled_parameters, list):
                for index, parameters in enumerate(scaled_parameters):
                    if "gamma" in parameters.kernel:
                        parameters.kernel["gamma"] *= (
                            initial_feature_counts[index]/current_feature_counts[index])
            else:
                if "gamma" in scaled_parameters.kernel:
                    scaled_parameters.kernel["gamma"] *= (
                        np.mean(initial_feature_counts)/np.mean(current_feature_counts))

            self._update_parameters(scaled_parameters)
            self._train_models()
            self._score_models()

        if self.separate_feature_sets:
            n_sets = len(self._active_perturbation_sets(0))
        else:
            n_sets = len(self._active_perturbation_sets())
            
        while(n_sets >= 2) :
            fit_current_feature_set()

            if self.separate_parameters:
                mean_performance = self.performance_[0]
                for m in range(1,self.num_models):
                    mean_performance = {key: mean_performance[key]+self.performance_[m][key] for key in self.performance_[m].keys()}
                row = dotdict({key: mean_performance[key]/self.num_models for key in mean_performance.keys()})    
            else:
                row = self.performance_
                
            if self.separate_feature_sets:
                row["num_features"] = np.sum([len(self.features[m]) for m in range(self.num_models)])/self.num_models
            else:
                row["num_features"] = len(self.features) 
                
            row["mean_nSV"] = np.sum(np.sum([self.models[m].n_support_ for m in range(self.num_models)]))/self.num_models
            print(f"Number of Features: {row['num_features']:.0f}, Score: {row['score']:.3f}")
            
            feature_performance[result] = row
            result += 1    

            if row.score >= best_score:
                best_models = copy.deepcopy(self.models)
                best_kernel_matrix = copy.deepcopy(self.kernel_matrix_)
                best_parameters = copy.deepcopy(self.parameters_)
                best_performance = copy.deepcopy(self.performance_)
                best_score = row.score
                best_features = copy.deepcopy(self.features)
            
            if self.separate_feature_sets:
                for i in range(self.num_models):
                    feature_rank = feature_ranker(self,i,set_for_rank)
                    active_sets = self._active_perturbation_sets(i)
                    n_to_remove = max(
                        1, min(len(active_sets) - 1,
                               int(np.floor(len(active_sets)*reduction_factor))))
                    selected_sets = np.argsort(feature_rank)[:n_to_remove]
                    to_remove = np.concatenate([
                        active_sets[index] for index in selected_sets])
                    self._remove_features(
                        to_remove, model_index = i, update_kernel = False)

                n_sets = len(self._active_perturbation_sets(0))
            else:
                active_sets = self._active_perturbation_sets()
                rank_total = np.zeros(len(active_sets))
                for i in range(self.num_models):
                    rank_total = rank_total + feature_ranker(self,i,set_for_rank)
                
                consensus_rank = rank_items(rank_total)
                n_to_remove = max(
                    1, min(len(active_sets) - 1,
                           int(np.floor(len(active_sets)*reduction_factor))))
                selected_sets = np.argsort(consensus_rank)[:n_to_remove]
                to_remove = np.concatenate([
                    active_sets[index] for index in selected_sets])
                self._remove_features(to_remove, update_kernel = False)
                n_sets = len(self._active_perturbation_sets())
        
        if n_sets > 0:
            fit_current_feature_set()
            
            if self.separate_parameters:
                mean_performance = self.performance_[0]
                for m in range(1,self.num_models):
                    mean_performance = {key: mean_performance[key]+self.performance_[m][key] for key in self.performance_[m].keys()}
                row = dotdict({key: mean_performance[key]/self.num_models for key in mean_performance.keys()})    
            else:
                row = self.performance_
            
            if self.separate_feature_sets:
                row["num_features"] = np.sum([len(self.features[m]) for m in range(self.num_models)])/self.num_models
            else:
                row["num_features"] = len(self.features) 
            
            row["mean_nSV"] = np.sum(np.sum([self.models[m].n_support_ for m in range(self.num_models)]))/self.num_models
            print(f"Number of Features: {row['num_features']:.0f}, Score: {row['score']:.3f}")
        
            feature_performance[result] = row
            
        if self.separate_feature_sets:
            self.sorted_features = []
            self.feature_rank = []
            for i in range(self.num_models):
                self.removed_features_[i] = np.append(self.removed_features_[i],self.features[i])
                self.sorted_features.append(np.flip(self.removed_features_[i]))
                self.feature_rank.append(self.sorted_features[i].argsort())
        else:
            self.removed_features_ = np.append(self.removed_features_,self.features)
            self.sorted_features =  np.flip(self.removed_features_)
            self.feature_rank =  self.sorted_features.argsort()    

        self.feature_performance_ = feature_performance

        self.models = best_models
        self.parameters_ = best_parameters
        self.performance_ = best_performance
        self.features = best_features
        self.kernel_matrix_ = best_kernel_matrix


    def greedy_forward_selection(self, parameter_grid,
                                 reduction_factor=0.1,
                                 feature_ranker=combined_rank().compute,
                                 set_for_rank="train",
                                 tune_models_each_step=True):
        """Rank and add feature sets using greedy forward selection.

        Every perturbation set is fitted by itself in the first round and the
        best singleton is retained.  Later rounds use the same perturbation
        ranker as backward selection, but perturb inactive sets by adding them;
        consequently forward decision perturbations have the opposite sign.
        """
        candidates = [np.asarray(group, dtype=int)
                      for group in self.perturbation_sets]
        if not candidates:
            raise ValueError("forward selection requires at least one feature set")

        feature_performance = {}
        singleton_performance = {}
        best_score = -1e12
        best_state = None
        previous_direction = getattr(self, "_selection_direction_", None)
        self._selection_direction_ = "forward"

        def mean_row():
            if self.separate_parameters:
                totals = copy.deepcopy(self.performance_[0])
                for model_index in range(1, self.num_models):
                    totals = {key: totals[key] + self.performance_[model_index][key]
                              for key in totals}
                return dotdict({key: totals[key] / self.num_models for key in totals})
            return copy.deepcopy(self.performance_)

        def save_state(row):
            nonlocal best_score, best_state
            if row.score > best_score:
                best_score = row.score
                best_state = (copy.deepcopy(self.models),
                              copy.deepcopy(self.kernel_matrix_),
                              copy.deepcopy(self.parameters_),
                              copy.deepcopy(self.performance_),
                              copy.deepcopy(self.features))

        try:
            # The first round is deliberately exhaustive rather than based on
            # a perturbation of an unfitted, zero-feature model.
            for candidate_index, candidate in enumerate(candidates):
                if self.separate_feature_sets:
                    for model_index in range(self.num_models):
                        self._set_features(candidate, model_index,
                                           update_kernel=False)
                else:
                    self._set_features(candidate, update_kernel=False)
                self.tune_models(parameter_grid)
                row = mean_row()
                singleton_performance[candidate_index] = copy.deepcopy(row)
                save_state(row)

            self.models, self.kernel_matrix_, self.parameters_, \
                self.performance_, self.features = copy.deepcopy(best_state)
            if self.separate_feature_sets:
                selection_order = [list(features) for features in self.features]
            else:
                selection_order = list(self.features)
            baseline_parameters = copy.deepcopy(self.parameters_)
            initial_count = np.mean([
                len(features) for features in self.features
            ]) if self.separate_feature_sets else len(self.features)

            result = 0
            while True:
                row = mean_row()
                row["num_features"] = (np.mean([len(features) for features in self.features])
                                       if self.separate_feature_sets
                                       else len(self.features))
                row["mean_nSV"] = np.sum([
                    np.sum(model.n_support_) for model in self.models
                ]) / self.num_models
                print(f"Number of Features: {row['num_features']:.0f}, "
                      f"Score: {row['score']:.3f}")
                feature_performance[result] = copy.deepcopy(row)
                result += 1
                save_state(row)

                inactive = self._inactive_perturbation_sets(
                    0 if self.separate_feature_sets else None)
                if not inactive:
                    break

                if self.separate_feature_sets:
                    for model_index in range(self.num_models):
                        model_inactive = self._inactive_perturbation_sets(model_index)
                        ranks = feature_ranker(self, model_index, set_for_rank)
                        n_to_add = max(1, min(len(model_inactive),
                            int(np.floor(len(model_inactive) * reduction_factor))))
                        chosen = np.argsort(ranks)[-n_to_add:]
                        additions = np.concatenate([
                            model_inactive[index] for index in chosen])
                        selection_order[model_index].extend(additions.tolist())
                        self._add_features(additions,
                            model_index, update_kernel=False)
                else:
                    rank_total = np.zeros(len(inactive))
                    for model_index in range(self.num_models):
                        rank_total += feature_ranker(self, model_index, set_for_rank)
                    consensus = rank_items(rank_total)
                    n_to_add = max(1, min(len(inactive),
                        int(np.floor(len(inactive) * reduction_factor))))
                    chosen = np.argsort(consensus)[-n_to_add:]
                    additions = np.concatenate([
                        inactive[index] for index in chosen])
                    selection_order.extend(additions.tolist())
                    self._add_features(additions, update_kernel=False)

                if tune_models_each_step:
                    self.tune_models(parameter_grid)
                else:
                    scaled = copy.deepcopy(baseline_parameters)
                    current_count = (np.mean([len(features) for features in self.features])
                                     if self.separate_feature_sets else len(self.features))
                    parameter_sets = scaled if isinstance(scaled, list) else [scaled]
                    for parameters in parameter_sets:
                        if "gamma" in parameters.kernel:
                            parameters.kernel["gamma"] *= initial_count/current_count
                    self._update_parameters(scaled)
                    self._train_models()
                    self._score_models()

            self.singleton_performance_ = singleton_performance
            self.feature_performance_ = feature_performance
            self.models, self.kernel_matrix_, self.parameters_, \
                self.performance_, self.features = best_state
            self._kernel_configuration_ = self._kernel_configuration(
                self.parameters_)
            if self.separate_feature_sets:
                self.sorted_features = [np.asarray(features)
                                        for features in selection_order]
                self.feature_rank = [features.argsort() for features in self.sorted_features]
            else:
                self.sorted_features = np.asarray(selection_order)
                self.feature_rank = self.sorted_features.argsort()
        finally:
            if previous_direction is None:
                self.__dict__.pop("_selection_direction_", None)
            else:
                self._selection_direction_ = previous_direction

    
    @staticmethod
    def _normalize_perturbation_sets(perturbation_sets, available_features):
        """Validate and normalize persistent feature perturbation groups."""
        available_features = set(np.asarray(available_features).tolist())
        normalized_sets = []
        for group in perturbation_sets:
            group = np.asarray(np.atleast_1d(group)).ravel()
            if group.size == 0:
                raise ValueError("perturbation_sets cannot contain an empty set")

            normalized_group = list(dict.fromkeys(group.tolist()))
            unknown_features = set(normalized_group).difference(available_features)
            if unknown_features:
                raise ValueError(
                    f"perturbation set contains unknown features: "
                    f"{sorted(unknown_features)}")
            normalized_sets.append(normalized_group)

        if not normalized_sets:
            raise ValueError("perturbation_sets must contain at least one set")

        flattened_features = [
            feature for group in normalized_sets for feature in group]
        if len(flattened_features) != len(set(flattened_features)):
            raise ValueError("perturbation_sets cannot overlap")
        missing_features = available_features.difference(flattened_features)
        if missing_features:
            raise ValueError(
                f"perturbation_sets must include every feature; missing: "
                f"{sorted(missing_features)}")
        return normalized_sets


    def _active_perturbation_sets(self, model_index = None):
        if self.separate_feature_sets:
            if model_index is None:
                raise ValueError("model_index is required for separate feature sets")
            current_features = self.features[model_index]
        else:
            current_features = self.features

        active_features = set(current_features.tolist())
        return [
            np.asarray([
                feature for feature in perturbation_set
                if feature in active_features])
            for perturbation_set in self.perturbation_sets
            if active_features.intersection(perturbation_set)]


    def _inactive_perturbation_sets(self, model_index=None):
        """Return complete perturbation sets absent from the current model."""
        if self.separate_feature_sets:
            if model_index is None:
                raise ValueError("model_index is required for separate feature sets")
            current_features = self.features[model_index]
        else:
            current_features = self.features
        active_features = set(np.asarray(current_features).tolist())
        return [np.asarray(group) for group in self.perturbation_sets
                if not active_features.intersection(group)]


    def feature_importance_(self, model_index):
        """Measure the effect of removing one or more feature groups.

        Parameters
        ----------
        model_index : int
            Index of the fitted cross-validation model to analyze.
        Perturbation groups are read from ``self.perturbation_sets``. Groups
        are intersected with the model's active features, and groups with no
        active members are ignored.
        """
        support_vectors = self._get_support_vectors(model_index)
        const = -0.5*(np.dot(self.models[model_index].dual_coef_[0,:],self.models[model_index].dual_coef_[0,:].transpose()))
        
        if self.separate_feature_sets:
            current_features = self.features[model_index]
        else:
            current_features = self.features

        if self.separate_parameters:
            parameters = self.parameters_[model_index].kernel
        else:
            parameters = self.parameters_.kernel

        forward = getattr(self, "_selection_direction_", None) == "forward"
        normalized_sets = (self._inactive_perturbation_sets(model_index)
                           if forward else
                           self._active_perturbation_sets(model_index))
        if not forward:
            for group in normalized_sets:
                if group.size == len(current_features):
                    raise ValueError("a perturbation set cannot remove all active features")

        K = self.kernel.compute(support_vectors, 
                                feature_index = current_features, 
                                parameters = parameters)

        criteria = np.zeros(len(normalized_sets))
        for z, perturbation_set in enumerate(normalized_sets):
            if forward:
                features_z = np.concatenate((current_features, perturbation_set))
            else:
                features_z = current_features[
                    ~np.isin(current_features, perturbation_set)]
            Kp = self.kernel.compute(support_vectors, 
                                     feature_index = features_z, 
                                     parameters = parameters)
            
            criteria[z] = np.sum(const*(K-Kp))
                
        return criteria

    
    def probability_perturbation_(self, model_index, X):
        probability = self.models[model_index].predict_proba(X)
        decision = self.models[model_index].decision_function(X)
        
        constant = -self.models[model_index].probA_*np.exp(self.models[model_index].probA_*decision \
                                                           + self.models[model_index].probB_)*probability**2
        
        decision_perturbation = self.decision_perturbation(model_index,X)
        probability_perturbation = decision_perturbation*constant
    
        return probability_perturbation

    
    def decision_perturbation_(self,model_index,X):
        support_vectors = self._get_support_vectors(model_index)

        if self.separate_feature_sets:
            current_features = self.features[model_index]
        else:
            current_features = self.features

        if self.separate_parameters:
            parameters = self.parameters_[model_index].kernel
        else:
            parameters = self.parameters_.kernel
        
        K = self.kernel.compute(X = support_vectors, 
                                feature_index = current_features, 
                                parameters = parameters, 
                                Y = X)
        
        forward = getattr(self, "_selection_direction_", None) == "forward"
        perturbation_sets = (self._inactive_perturbation_sets(model_index)
                             if forward else
                             self._active_perturbation_sets(model_index))
        decision_perturbation = np.zeros([len(X), len(perturbation_sets)])
        for z, perturbation_set in enumerate(perturbation_sets):
            if forward:
                features_z = np.concatenate((current_features, perturbation_set))
            else:
                features_z = current_features[
                    ~np.isin(current_features, perturbation_set)]
            if not forward and len(features_z) == 0:
                raise ValueError("a perturbation set cannot remove all active features")
            Kp = self.kernel.compute(support_vectors, 
                                     feature_index = features_z, 
                                     parameters = parameters, 
                                     Y = X)
            
            # K has shape (n_support, n_samples). A vector-matrix product
            # performs the support-vector reduction directly, avoiding a
            # tiled coefficient matrix and an equally large product array.
            decision_perturbation[:,z] = np.matmul(
                self.models[model_index].dual_coef_[0, :], K - Kp)
                        
        return decision_perturbation

            
    def decision_gradient_(self,model_index,X):
        support_vectors = self._get_support_vectors(model_index)

        if self.separate_feature_sets:
            current_features = self.features[model_index]
        else:
            current_features = self.features

        if self.separate_parameters:
            parameters = self.parameters_[model_index].kernel
        else:
            parameters = self.parameters_.kernel
            
        decision_gradient = np.zeros([len(X), len(current_features)])
        for j in range(0,len(current_features)):
            z = current_features[j]
            
            dK = self.kernel.compute_gradient(support_vectors, 
                                              feature_index = current_features,
                                              wrt = z,
                                              parameters = parameters,
                                              Y = X)    
            
            decision_gradient[:,j] = np.matmul(
                self.models[model_index].dual_coef_[0, :], dK)

        return decision_gradient

    
    def _find_boundary_points(self, model_index, X):
        boundary_points = np.zeros([len(X), self.cv.X.shape[1]])
        for i in range(0,len(X)):
            opt = minimize(svc_dec2, X[i,:], args=(self,model_index))
            boundary_points[i,:] = opt.x
            
        return boundary_points

    
    def integrated_gradient(self, X, model_index = None, num_steps = 20,
                            reference_point = None, ref_point = None):
        """Calculate integrated gradients from supplied or inferred references.

        ``reference_point`` may be one feature vector shared by every sample,
        or an array with one reference vector per row of ``X``. Supplying it
        bypasses decision-boundary optimization entirely. ``ref_point`` is
        retained as a backward-compatible alias.
        """
        if reference_point is not None and ref_point is not None:
            raise ValueError("specify only reference_point, not both aliases")
        if reference_point is None:
            reference_point = ref_point

        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.cv.X.shape[1]:
            raise ValueError("X must have shape (n_samples, n_features)")

        supplied_reference = reference_point is not None
        if supplied_reference:
            reference_point = np.asarray(reference_point)
            if reference_point.ndim == 1:
                if reference_point.shape[0] != X.shape[1]:
                    raise ValueError("reference_point must match X's feature count")
                reference_points = np.broadcast_to(reference_point, X.shape)
            elif reference_point.shape == X.shape:
                reference_points = reference_point
            else:
                raise ValueError(
                    "reference_point must have shape (n_features,) or match X")

        if isinstance(self.SVM, SVR) and not supplied_reference:
            raise NameError('SVRneedsRefPoint')

        if self.separate_feature_sets & (model_index is None):
            raise NameError('SeparateNeedsModelIndex')
        
        if model_index == None:
            model_indices = [i for i in range(self.num_models)]
        else:
            model_indices = [model_index]
            
        if self.separate_feature_sets:
            features = self.features[model_index]                                            
        else:
            features = self.features
            
        integrated_gradient = np.zeros([len(X), len(features)])    
        for m in model_indices:
            if supplied_reference:
                model_reference_points = reference_points
            else:
                model_reference_points = self._find_boundary_points(m,X)
        
            for i in range(0,len(X)):
                x_start = model_reference_points[i, :]
                
                xi = X[i,:]
                x_diff = xi - x_start
                
                path_fraction = np.linspace(0, 1, num_steps)[:, np.newaxis]
                x_steps = x_start + path_fraction*x_diff

                gradient_steps = self.decision_gradient_(m,x_steps)
                if isinstance(self.SVM, SVR):
                    ref_val = self.predict(x_start.reshape((1,-1)),model_index=m)
                    integrated_gradient[i,:] += (x_diff[features]*[np.trapz(gradient_steps[:,n])/ \
                                                   num_steps for n in range(gradient_steps.shape[1])] + ref_val)/len(features)
                else:
                    ref_val = self.decision_function(x_start.reshape((1,-1)),model_index=m)
                    #integrated_gradient[i,:] += x_diff[features]*[np.trapz(gradient_steps[:,n])/ \
                    #                              num_steps for n in range(gradient_steps.shape[1])]
                    integrated_gradient[i,:] += x_diff[features]*[np.trapz(gradient_steps[:,n])/ \
                                                   num_steps for n in range(gradient_steps.shape[1])]+ref_val/len(features)

                

        return integrated_gradient/len(model_indices)

    
    def plot_performance(self,metric = 'score'):
        x = [self.feature_performance_[key]['num_features'] for key in self.feature_performance_.keys()]
        y = [self.feature_performance_[key][metric] for key in self.feature_performance_.keys()]
        
        plt.plot(x,y)
        plt.xlabel('# of features') 
        plt.ylabel(metric) 


    def predict(self, X, model_index = None, use_voting = False):
        if isinstance(self.SVM, SVR):
            if model_index == None:
                model_indices = [i for i in range(self.num_models)]
            else:
                model_indices = [model_index]
    
            predictions = 0
            for m in model_indices:
                if self.separate_feature_sets:
                    feature_index = self.features[m]                                            
                else:
                    feature_index = self.features

                if self.separate_parameters:
                    parameters = self.parameters_[m].kernel                                            
                else:
                    parameters = self.parameters_.kernel
                    
                kernel_matrix = self.kernel.compute(X, 
                                                    feature_index = feature_index, 
                                                    parameters = parameters,
                                                    Y = self.cv.X[self.X_ind[m],:])
                
                predictions += self.models[m].predict(kernel_matrix)
            
            predictions = predictions/len(model_indices)

        else:
            if use_voting:
                if model_index == None:
                    model_indices = [i for i in range(self.num_models)]
                else:
                    model_indices = [model_index]
        
                positive_class = self.models[0].classes_[0]
                prediction_counts = 0
                for m in model_indices:
                    if self.separate_feature_sets:
                        feature_index = self.features[m]                                            
                    else:
                        feature_index = self.features
    
                    if self.separate_parameters:
                        parameters = self.parameters_[m].kernel                                            
                    else:
                        parameters = self.parameters_.kernel
                        
                    kernel_matrix = self.kernel.compute(X, 
                                                        feature_index = feature_index, 
                                                        parameters = parameters,
                                                        Y = self.cv.X[self.X_ind[m],:])
                    
                    model_predictions = self.models[m].predict(kernel_matrix)
                    prediction_counts += (model_predictions == positive_class) + 0
                
                predictions = self.models[0].classes_[(prediction_counts/len(model_indices) < 0.5) + 0] 
                
            else:
                decision_values = self.decision_function(X, model_index)
                predictions = self.models[0].classes_[(decision_values > 0) + 0]

        return predictions
        

    def decision_function(self, X, model_index = None):     
        if model_index == None:
            model_indices = [i for i in range(self.num_models)]
        else:
            model_indices = [model_index]

        decision_values = 0
        for m in model_indices:
            if self.separate_feature_sets:
                feature_index = self.features[m]                                            
            else:
                feature_index = self.features

            if self.separate_parameters:
                parameters = self.parameters_[m].kernel                                            
            else:
                parameters = self.parameters_.kernel
                    
            kernel_matrix = self.kernel.compute(X, 
                                                feature_index = feature_index, 
                                                parameters = parameters,
                                                Y = self.cv.X[self.X_ind[m],:])
            
            decision_values += self.models[m].decision_function(kernel_matrix)

        return decision_values/len(model_indices)


    def enrichment_score(self,metric = 'score',type = 'auc'):
        enrichment_score = []
        
        match type:
            case "auc":
                points = sorted(
                    ((row['num_features'], row[metric])
                     for row in self.feature_performance_.values()),
                    reverse=True)
                x, y = zip(*points)
                
                area = np.trapz(y,x)
                enrichment_score = -area/max(x)
            
            case "max":
                y = [self.feature_performance_[key][metric] for key in self.feature_performance_.keys()]
                enrichment_score = max(y)
                
        return enrichment_score
        
