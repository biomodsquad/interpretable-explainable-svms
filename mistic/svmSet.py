import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   

from scipy.optimize import minimize

import copy
from collections import Counter

from sklearn.svm import OneClassSVM, SVR
from sklearn.metrics import f1_score, roc_auc_score, r2_score
from sklearn.base import clone
from scipy.stats import pearsonr

import random
import warnings

from mistic.utility import combined_rank, kernelWrapper, score_svr, score_svc, dotdict, svc_dec2, rank_items
from mistic.explanations import IntegratedGradientsResult

class svmSet():

    def _is_one_class(self):
        return isinstance(self.SVM, OneClassSVM)
        
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
        self._update_unified_feature_attributes()

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
        self.unified_model_ = None
        self.unified_parameters_ = None
        self.unified_prediction_features_ = None
        # Binary classifiers replace this default with an out-of-fold F1
        # optimum after tuning. Keeping the default makes an untuned ensemble
        # and older serialized ensembles behave as they did previously.
        self.decision_value_cutoff_ = 0.0
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
            if "decision_value_cutoff_" not in self.__dict__:
                self.decision_value_cutoff_ = 0.0
            if "unified_model_" not in self.__dict__:
                self.unified_model_ = None
                self.unified_parameters_ = None
                self.unified_prediction_features_ = None
            self._update_unified_feature_attributes()

    def _update_unified_feature_attributes(self):
        """Refresh ensemble-level feature membership and rank attributes."""
        if self.separate_feature_sets:
            feature_sets = [np.asarray(features, dtype=int)
                            for features in self.features]
            self.unified_features = np.unique(np.concatenate(feature_sets))
        else:
            self.unified_features = np.asarray(self.features, dtype=int).copy()

        if not hasattr(self, "feature_rank"):
            self.unified_feature_rank = None
            self.unified_sorted_features = None
            return

        if self.separate_feature_sets:
            ranks = np.asarray(self.feature_rank, dtype=float)
            self.unified_feature_rank = np.mean(ranks, axis=0)
        else:
            self.unified_feature_rank = np.asarray(
                self.feature_rank, dtype=float).copy()
        self.unified_sorted_features = np.argsort(
            self.unified_feature_rank, kind="stable")
    
    def _train_models(self):
        for i in range(self.num_models):
            if self.separate_feature_sets | self.separate_parameters:
                kernel_matrix = self._get_kernel_matrix(self.X_ind[i],self.X_ind[i],model_index = i)
            else:
                kernel_matrix = self._get_kernel_matrix(self.X_ind[i],self.X_ind[i])
                
            if self._is_one_class():
                self.models[i].fit(kernel_matrix)
            else:
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

    def mean_performance(self):
        """Return performance averaged across cross-validation models.

        When ``performance_`` is a list of per-model mappings, numeric scalar
        values are averaged across all models. Non-numeric values are retained
        only when they are identical in every model. If ``performance_`` is
        already an aggregate mapping, a copy is returned unchanged.

        Returns
        -------
        mistic.utility.dotdict
            Aggregate performance values with attribute-style access.

        Raises
        ------
        RuntimeError
            If models have not been scored yet.
        TypeError
            If ``performance_`` is neither a mapping nor a non-empty sequence
            of mappings.
        """
        if not hasattr(self, "performance_"):
            raise RuntimeError("models must be scored before summarizing performance")

        if isinstance(self.performance_, dict):
            return dotdict(copy.deepcopy(self.performance_))

        rows = self.performance_
        if (not isinstance(rows, (list, tuple)) or not rows or
                not all(isinstance(row, dict) for row in rows)):
            raise TypeError(
                "performance_ must be a mapping or a non-empty sequence of mappings")

        summary = dotdict()
        common_keys = set(rows[0]).intersection(*(set(row) for row in rows[1:]))
        for key in rows[0]:
            if key not in common_keys:
                continue
            values = [row[key] for row in rows]
            arrays = [np.asarray(value) for value in values]
            if all(array.ndim == 0 and np.issubdtype(array.dtype, np.number)
                   for array in arrays):
                summary[key] = float(np.mean([array.item() for array in arrays]))
            elif all(value == values[0] for value in values[1:]):
                summary[key] = copy.deepcopy(values[0])
        return summary

    
    def tune_models(self, parameter_grid):
        parameter_grid = list(parameter_grid)
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
        if (not isinstance(self.SVM, SVR) and not self._is_one_class() and
                np.asarray(self.models[0].classes_).size == 2):
            self.calibrate_decision_value_cutoff()
        if not getattr(self, "_defer_unified_fit_", False):
            self.fit_unified_model(parameter_grid)


    def _tune_member_models(self, parameter_grid):
        """Tune member models without fitting an intermediate unified model."""
        previous = getattr(self, "_defer_unified_fit_", False)
        self._defer_unified_fit_ = True
        try:
            self.tune_models(parameter_grid)
        finally:
            self._defer_unified_fit_ = previous


    def _ranked_unified_prediction_features(self):
        """Return the knee-limited unified feature ranking for prediction."""
        unified = set(np.asarray(self.unified_features, dtype=int))
        ranking = getattr(self, "unified_sorted_features", None)
        if ranking is None:
            ranked = np.asarray(sorted(unified), dtype=int)
        else:
            ranked = np.asarray(
                [feature for feature in ranking if feature in unified], dtype=int)
        limit = getattr(self, "knee_num_features_", len(ranked))
        return ranked[:min(int(limit), len(ranked))]


    def _score_unified_fold(self, model, kernel_matrix, y_true):
        """Apply the configured MiSTIC score to a unified validation fold."""
        predictions = model.predict(kernel_matrix)
        if self._is_one_class():
            labels = np.unique(y_true)
            if not np.all(np.isin(labels, [-1, 1])):
                raise ValueError("one-class validation labels must be -1 or +1")
            if labels.size == 1:
                return float(np.mean(predictions == 1))
            f1 = f1_score(y_true, predictions, pos_label=1)
            auc = roc_auc_score(y_true, model.decision_function(kernel_matrix))
            weight = getattr(getattr(self.score, "__self__", None), "weight", 0.5)
            return weight * auc + (1-weight) * f1
        if isinstance(self.SVM, SVR):
            if len(np.unique(predictions)) <= 2:
                pearson = 0.00001
                coefficient = 0.00001
            else:
                pearson = pearsonr(y_true, predictions).statistic ** 2
                coefficient = r2_score(y_true, predictions)
            weight = getattr(getattr(self.score, "__self__", None), "weight", 0.5)
            return weight * float(pearson) + (1-weight) * max(0.00001, coefficient)

        classes = np.asarray(model.classes_)
        if classes.size != 2:
            return float(np.mean(predictions == y_true))
        positive = classes[1]
        f1 = f1_score(y_true, predictions, pos_label=positive)
        auc = roc_auc_score(y_true, model.decision_function(kernel_matrix))
        weight = getattr(getattr(self.score, "__self__", None), "weight", 0.5)
        return weight * auc + (1-weight) * f1


    def fit_unified_model(self, parameter_grid):
        """Tune and fit one SVM on the knee-ranked unified feature subset.

        Candidate parameters are evaluated with the existing ``cvSet``
        splits. The winning model is then fitted once on all labeled samples
        supplied to the ``cvSet``. Member models remain unchanged and are
        still available through ``prediction_mode='set'`` or ``model_index``.
        """
        parameter_grid = list(parameter_grid)
        if not parameter_grid:
            raise ValueError("parameter_grid must contain at least one candidate")
        features = self._ranked_unified_prediction_features()
        if not len(features):
            raise RuntimeError("the unified predictor requires selected features")

        best_score = -np.inf
        best_parameters = None
        for parameters in parameter_grid:
            fold_scores = []
            for train_indices, test_indices in zip(self.cv.train, self.cv.test):
                train_indices = np.asarray(train_indices, dtype=int)
                test_indices = np.asarray(test_indices, dtype=int)
                model = clone(self.SVM).set_params(**parameters.model)
                train_kernel = self.kernel.compute(
                    self.cv.X[train_indices], feature_index=features,
                    parameters=parameters.kernel,
                    Y=self.cv.X[train_indices])
                if self._is_one_class():
                    model.fit(train_kernel)
                else:
                    model.fit(train_kernel, self.cv.y[train_indices])
                test_kernel = self.kernel.compute(
                    self.cv.X[test_indices], feature_index=features,
                    parameters=parameters.kernel,
                    Y=self.cv.X[train_indices])
                fold_scores.append(self._score_unified_fold(
                    model, test_kernel, self.cv.y[test_indices]))
            score = float(np.mean(fold_scores))
            if score > best_score:
                best_score = score
                best_parameters = copy.deepcopy(parameters)

        training_indices = (np.flatnonzero(self.cv.y == 1)
                            if self._is_one_class() and
                            getattr(self.cv, "type", None) == "one-class"
                            else np.arange(self.num_samples))
        model = clone(self.SVM).set_params(**best_parameters.model)
        # The precomputed training kernel must be square over the observations
        # used for fitting.
        training_kernel = self.kernel.compute(
            self.cv.X[training_indices], feature_index=features,
            parameters=best_parameters.kernel,
            Y=self.cv.X[training_indices])
        if self._is_one_class():
            model.fit(training_kernel)
        else:
            model.fit(training_kernel, self.cv.y[training_indices])
        self.unified_model_ = model
        self.unified_parameters_ = best_parameters
        self.unified_prediction_features_ = features
        self.unified_training_indices_ = training_indices
        self.unified_cv_score_ = best_score
        return self


    @staticmethod
    def _optimal_f1_cutoff(decision_values, y_true, positive_class):
        """Return the ``decision_value > cutoff`` threshold maximizing F1."""
        decision_values = np.asarray(decision_values, dtype=float).ravel()
        y_positive = np.asarray(y_true).ravel() == positive_class
        if decision_values.size == 0:
            raise ValueError("at least one decision value is required")
        if decision_values.size != y_positive.size:
            raise ValueError("decision values and labels must have equal length")
        if not np.all(np.isfinite(decision_values)):
            raise ValueError("decision values must be finite")

        # With a strict `>` comparison, a cutoff immediately below the
        # minimum covers the all-positive prediction, and each observed value
        # covers every other distinct prediction partition.
        candidates = np.concatenate((
            [np.nextafter(np.min(decision_values), -np.inf)],
            np.unique(decision_values),
        ))
        scores = np.empty(len(candidates), dtype=float)
        for index, cutoff in enumerate(candidates):
            predicted_positive = decision_values > cutoff
            true_positive = np.count_nonzero(predicted_positive & y_positive)
            false_positive = np.count_nonzero(predicted_positive & ~y_positive)
            false_negative = np.count_nonzero(~predicted_positive & y_positive)
            denominator = 2 * true_positive + false_positive + false_negative
            scores[index] = (2 * true_positive / denominator
                             if denominator else 0.0)

        best = np.flatnonzero(scores == np.max(scores))
        # Prefer the least disruptive threshold when several cutoffs produce
        # the same optimum.
        return float(candidates[best[np.argmin(np.abs(candidates[best]))]])


    def calibrate_decision_value_cutoff(self):
        """Calibrate the binary ``svmSet`` cutoff on CV development data.

        The decision value for each sample is averaged across all fitted SVMs,
        exactly as it is during aggregate :meth:`predict` inference. The
        resulting cutoff therefore corrects a shift introduced by averaging
        the fold models' decision values. Any reserved ensemble-validation
        samples are excluded from this calibration.

        Returns
        -------
        float
            The F1-optimal cutoff, also stored in
            ``decision_value_cutoff_``.
        """
        classes = np.asarray(self.models[0].classes_)
        if classes.size != 2:
            raise ValueError(
                "decision-value cutoff calibration requires binary classification")

        calibration_indices = getattr(
            self.cv, "development_indices_", np.arange(len(self.cv.y)))
        self.decision_value_cutoff_ = self._optimal_f1_cutoff(
            self._set_decision_function(self.cv.X[calibration_indices]),
            self.cv.y[calibration_indices],
            positive_class=classes[1],
        )
        return self.decision_value_cutoff_

    
    def _get_support_vectors(self,model_index):
        return self.cv.X[self.X_ind[model_index],:][self.models[model_index].support_,:]


    def _inference_kernel(self, X, model_index):
        """Build a precomputed kernel while evaluating support vectors only."""
        if self.separate_feature_sets:
            feature_index = self.features[model_index]
        else:
            feature_index = self.features

        if self.separate_parameters:
            parameters = self.parameters_[model_index].kernel
        else:
            parameters = self.parameters_.kernel

        support_positions = self.models[model_index].support_
        support_indices = self.X_ind[model_index][support_positions]
        support_kernel = self.kernel.compute(
            X,
            feature_index=feature_index,
            parameters=parameters,
            Y=self.cv.X[support_indices, :])

        # Precomputed-kernel estimators validate against the number of rows
        # used during fitting. Preserve that width, filling only the columns
        # libsvm will read for the fitted support vectors.
        kernel_matrix = np.zeros(
            (len(X), len(self.X_ind[model_index])),
            dtype=support_kernel.dtype)
        kernel_matrix[:, support_positions] = support_kernel
        return kernel_matrix


    def _unified_inference_kernel(self, X):
        if self.unified_model_ is None:
            raise RuntimeError(
                "the unified predictor has not been fitted; call "
                "fit_unified_model or tune_models first")
        support_positions = self.unified_model_.support_
        support_kernel = self.kernel.compute(
            X, feature_index=self.unified_prediction_features_,
            parameters=self.unified_parameters_.kernel,
            Y=self.cv.X[self.unified_training_indices_[support_positions]])
        kernel_matrix = np.zeros(
            (len(X), len(self.unified_training_indices_)),
            dtype=support_kernel.dtype)
        kernel_matrix[:, support_positions] = support_kernel
        return kernel_matrix
        

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
        self.unified_model_ = None
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
        self.unified_model_ = None
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
        self._update_unified_feature_attributes()
            
        if len(current_features) > 0 and update_kernel:
            self._update_kernel_matrix()
            self._kernel_configuration_ = self._kernel_configuration(self.parameters_)


    def _set_features(self, features, model_index=None, update_kernel=True):
        """Replace the active feature set while preserving its original order."""
        self.unified_model_ = None
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
        self._update_unified_feature_attributes()

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


    def stochastic_feature_selection(self, parameter_grid, n_iterations=100,
                                     feature_ranker=combined_rank().compute,
                                     set_for_rank="train", temperature=0.05,
                                     cooling_rate=0.97, add_probability=0.5,
                                     random_seed=None,
                                     update_all_models=False,
                                     use_ensemble_validation=False,
                                     expected_changes_per_model=1.0,
                                     preserve_feature_count=False,
                                     convergence_patience=20,
                                     convergence_min_delta=0.0):
        """Refine the current feature set with rank-guided stochastic moves.

        Each iteration proposes adding or removing one complete perturbation
        set. Candidate probabilities are biased by ``feature_ranker`` in the
        same direction as the greedy searches: highly ranked inactive sets
        are preferred for addition and low-ranked active sets for removal.
        Improving moves are always accepted; other moves are accepted with a
        simulated-annealing probability.

        The search starts from the current fitted feature set, so it is most
        useful after :meth:`greedy_forward_selection` or
        :meth:`greedy_backward_selection`. With separate feature sets, a move
        changes one model's set; otherwise it changes the ensemble-wide set.
        Every proposal is tuned using ``parameter_grid``. The best accepted
        state, rather than merely the last state, is restored on return. If
        ``update_all_models`` is true for an ensemble with separate feature
        sets, every model proposes one independently ranked add/remove move
        per iteration. Those moves form one joint proposal and are accepted
        or rejected together according to the average cross-validation score.
        ``expected_changes_per_model`` controls how many perturbation groups
        use the chosen add/remove operation for each selected model. The count
        is sampled as ``1 + Poisson(expected_changes_per_model - 1)`` and
        capped by the number of feasible groups, so every eligible selected
        model changes at least once. The default of one preserves single-group
        proposals exactly.
        When ``preserve_feature_count`` is true, each model instead proposes
        one removal and one addition of equally sized perturbation groups.
        Separate-feature-set ensembles update every model in this mode, and
        the complete collection of swaps is accepted or rejected jointly.
        If ``use_ensemble_validation`` is true, acceptance and best-state
        tracking instead use the mean model score on the holdout reserved by
        :class:`cvSet`; parameter tuning still uses the ordinary CV folds.
        The search converges early after ``convergence_patience`` consecutive
        proposals fail to improve the best accepted objective by more than
        ``convergence_min_delta``. Set patience to ``None`` to always run the
        requested number of iterations.

        Search diagnostics are stored in ``stochastic_performance_``. Each
        row records the iteration, operation, model index, proposed group,
        score, acceptance, temperature, and resulting feature membership.

        Returns
        -------
        svmSet
            The refined ensemble (``self``).
        """
        if (isinstance(n_iterations, (bool, np.bool_)) or
                not isinstance(n_iterations, (int, np.integer))):
            raise TypeError("n_iterations must be an integer")
        if n_iterations < 1:
            raise ValueError("n_iterations must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if not 0 < cooling_rate <= 1:
            raise ValueError("cooling_rate must be in (0, 1]")
        if not 0 <= add_probability <= 1:
            raise ValueError("add_probability must be between 0 and 1")
        if not isinstance(update_all_models, (bool, np.bool_)):
            raise TypeError("update_all_models must be boolean")
        if not isinstance(use_ensemble_validation, (bool, np.bool_)):
            raise TypeError("use_ensemble_validation must be boolean")
        if not isinstance(preserve_feature_count, (bool, np.bool_)):
            raise TypeError("preserve_feature_count must be boolean")
        if (isinstance(expected_changes_per_model, (bool, np.bool_)) or
                not isinstance(expected_changes_per_model,
                               (int, float, np.integer, np.floating))):
            raise TypeError("expected_changes_per_model must be numeric")
        expected_changes_per_model = float(expected_changes_per_model)
        if (not np.isfinite(expected_changes_per_model) or
                expected_changes_per_model < 1):
            raise ValueError(
                "expected_changes_per_model must be finite and at least 1")
        if preserve_feature_count and expected_changes_per_model != 1:
            raise ValueError(
                "expected_changes_per_model must equal 1 when "
                "preserve_feature_count is enabled")
        if convergence_patience is not None:
            if (isinstance(convergence_patience, (bool, np.bool_)) or
                    not isinstance(convergence_patience, (int, np.integer))):
                raise TypeError("convergence_patience must be an integer or None")
            if convergence_patience < 1:
                raise ValueError("convergence_patience must be at least 1")
        if (isinstance(convergence_min_delta, (bool, np.bool_)) or
                not isinstance(convergence_min_delta,
                               (int, float, np.integer, np.floating))):
            raise TypeError("convergence_min_delta must be numeric")
        convergence_min_delta = float(convergence_min_delta)
        if not np.isfinite(convergence_min_delta) or convergence_min_delta < 0:
            raise ValueError(
                "convergence_min_delta must be finite and non-negative")
        validation_indices = getattr(
            self.cv, "ensemble_validation_indices_", np.array([], dtype=int))
        if use_ensemble_validation and len(validation_indices) == 0:
            raise ValueError(
                "use_ensemble_validation requires a non-empty ensemble "
                "validation set in cvSet")
        if not hasattr(self, "parameters_") or not self.models:
            raise RuntimeError(
                "models must be fitted before stochastic feature selection")

        parameter_grid = list(parameter_grid)
        if not parameter_grid:
            raise ValueError("parameter_grid must contain at least one candidate")
        rng = np.random.default_rng(random_seed)

        def mean_score():
            if use_ensemble_validation:
                scores = []
                for model_index in range(self.num_models):
                    original_test = self.cv.test[model_index]
                    try:
                        self.cv.test[model_index] = validation_indices
                        scores.append(float(
                            self.score(self, model_index=model_index)["score"]))
                    finally:
                        self.cv.test[model_index] = original_test
                return float(np.mean(scores))
            if self.separate_parameters:
                return float(np.mean([
                    performance.score for performance in self.performance_]))
            return float(self.performance_.score)

        def snapshot():
            return copy.deepcopy((
                self.models, self.kernel_matrix_, self.parameters_,
                self.performance_, self.features,
                getattr(self, "decision_value_cutoff_", 0.0),
                self._kernel_configuration_, self.removed_features_,
            ))

        def restore(state):
            (self.models, self.kernel_matrix_, self.parameters_,
             self.performance_, self.features,
             self.decision_value_cutoff_,
             self._kernel_configuration_,
             self.removed_features_) = copy.deepcopy(state)
            self._update_unified_feature_attributes()

        def weighted_choices(ranks, prefer_high, count):
            ranks = np.asarray(ranks, dtype=float)
            desirability = ranks if prefer_high else np.max(ranks) - ranks
            # A positive offset keeps every move reachable and avoids a
            # deterministic search when rank values contain zero.
            weights = desirability + 1.0
            return np.atleast_1d(rng.choice(
                len(ranks), size=count, replace=False,
                p=weights / np.sum(weights))).astype(int)

        def ranked_groups(operation, model_index):
            previous_direction = getattr(self, "_selection_direction_", None)
            if operation == "add":
                self._selection_direction_ = "forward"
                groups = self._inactive_perturbation_sets(model_index)
            else:
                self.__dict__.pop("_selection_direction_", None)
                groups = self._active_perturbation_sets(model_index)
            try:
                if self.separate_feature_sets:
                    ranks = feature_ranker(self, model_index, set_for_rank)
                else:
                    rank_total = np.zeros(len(groups))
                    for index in range(self.num_models):
                        rank_total += feature_ranker(self, index, set_for_rank)
                    ranks = rank_items(rank_total)
            finally:
                if previous_direction is None:
                    self.__dict__.pop("_selection_direction_", None)
                else:
                    self._selection_direction_ = previous_direction
            return groups, np.asarray(ranks)

        def feasible_operations(model_index):
            operations = []
            if self._inactive_perturbation_sets(model_index):
                operations.append("add")
            if len(self._active_perturbation_sets(model_index)) > 1:
                operations.append("remove")
            return operations

        def can_swap(model_index):
            active = self._active_perturbation_sets(model_index)
            inactive = self._inactive_perturbation_sets(model_index)
            return (len(active) > 1 and any(
                len(active_group) == len(inactive_group)
                for active_group in active for inactive_group in inactive))

        def propose_swap(model_index):
            if not can_swap(model_index):
                return []
            active, remove_ranks = ranked_groups("remove", model_index)
            inactive, add_ranks = ranked_groups("add", model_index)
            removable_indices = [index for index, group in enumerate(active)
                                 if any(len(group) == len(candidate)
                                        for candidate in inactive)]
            selected_remove = int(weighted_choices(
                remove_ranks[removable_indices], prefer_high=False,
                count=1)[0])
            remove_index = removable_indices[selected_remove]
            addable_indices = [index for index, group in enumerate(inactive)
                               if len(group) == len(active[remove_index])]
            selected_add = int(weighted_choices(
                add_ranks[addable_indices], prefer_high=True, count=1)[0])
            add_index = addable_indices[selected_add]
            return [
                ("remove", model_index, active[remove_index]),
                ("add", model_index, inactive[add_index]),
            ]

        def propose_moves(model_index):
            operations = feasible_operations(model_index)
            if not operations:
                return []
            choose_add = bool(rng.random() < add_probability)
            operation = "add" if choose_add and "add" in operations else "remove"
            if operation not in operations:
                operation = "add"
            groups, ranks = ranked_groups(operation, model_index)
            requested_count = 1 + int(rng.poisson(
                expected_changes_per_model - 1.0))
            max_changes = (len(groups) if operation == "add"
                           else len(groups) - 1)
            change_count = min(requested_count, max_changes)
            group_indices = weighted_choices(
                ranks, prefer_high=operation == "add", count=change_count)
            return [(operation, model_index, groups[group_index])
                    for group_index in group_indices]

        current_score = mean_score()
        current_state = snapshot()
        best_score = current_score
        best_state = snapshot()
        convergence_score = current_score
        iterations_without_improvement = 0
        converged = False
        stop_reason = "max_iterations"
        history = []
        current_temperature = float(temperature)

        for iteration in range(int(n_iterations)):
            if preserve_feature_count:
                targets = (range(self.num_models)
                           if self.separate_feature_sets else [None])
                swap_batches = [propose_swap(model_index)
                                for model_index in targets]
                moves = ([] if any(not batch for batch in swap_batches)
                         else [move for batch in swap_batches for move in batch])
            elif self.separate_feature_sets and update_all_models:
                moves = [move for model_index in range(self.num_models)
                         for move in propose_moves(model_index)]
            else:
                targets = (list(range(self.num_models))
                           if self.separate_feature_sets else [None])
                eligible_targets = [model_index for model_index in targets
                                    if feasible_operations(model_index)]
                moves = ([] if not eligible_targets else propose_moves(
                    eligible_targets[int(rng.integers(len(eligible_targets)))])
                )
            if not moves:
                stop_reason = "no_feasible_moves"
                break

            for operation, model_index, group in moves:
                if operation == "add":
                    self._add_features(group, model_index=model_index,
                                       update_kernel=False)
                else:
                    self._remove_features(group, model_index=model_index,
                                          update_kernel=False)
            self._tune_member_models(parameter_grid)
            proposed_score = mean_score()
            score_change = proposed_score - current_score
            accepted = (score_change >= 0 or rng.random() <
                        np.exp(score_change / current_temperature))
            if accepted:
                current_score = proposed_score
                current_state = snapshot()
                if proposed_score > best_score:
                    best_score = proposed_score
                    best_state = snapshot()
                if proposed_score > convergence_score + convergence_min_delta:
                    convergence_score = proposed_score
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            else:
                restore(current_state)
                iterations_without_improvement += 1

            operations = [move[0] for move in moves]
            model_indices = [move[1] for move in moves]
            changed_groups = [move[2].copy() for move in moves]
            changes_by_model = [
                sum(move[1] == model_index for move in moves)
                for model_index in (range(self.num_models)
                                    if self.separate_feature_sets else [None])]
            history.append(dotdict({
                "iteration": iteration,
                "operation": (operations[0] if len(moves) == 1 else operations),
                "model_index": (model_indices[0]
                                if len(moves) == 1 else model_indices),
                "features_changed": (changed_groups[0]
                                     if len(moves) == 1 else changed_groups),
                "moves": [dotdict({
                    "operation": operation,
                    "model_index": model_index,
                    "features_changed": group.copy(),
                }) for operation, model_index, group in moves],
                "num_changes": len(moves),
                "num_changes_by_model": changes_by_model,
                "score": proposed_score,
                "objective": ("ensemble_validation" if use_ensemble_validation
                              else "cross_validation"),
                "accepted": accepted,
                "temperature": current_temperature,
                "iterations_without_improvement":
                    iterations_without_improvement,
                "features": copy.deepcopy(self.features),
            }))
            current_temperature *= cooling_rate
            if (convergence_patience is not None and
                    iterations_without_improvement >= convergence_patience):
                converged = True
                stop_reason = "converged"
                break

        restore(best_state)
        self.stochastic_performance_ = history
        self.stochastic_best_score_ = best_score
        self.stochastic_converged_ = converged
        self.stochastic_iterations_ = len(history)
        self.stochastic_stop_reason_ = stop_reason
        self.fit_unified_model(parameter_grid)
        return self


    def ensemble_stochastic_feature_selection(
            self, parameter_grid, n_iterations=100, temperature=0.05,
            cooling_rate=0.97, add_probability=0.5, random_seed=None,
            convergence_patience=20, convergence_min_delta=0.0,
            preserve_feature_count=False, feature_diversity_weight=0.0,
            prediction_diversity_weight=0.0,
            performance_tolerance=None, max_feature_similarity=None):
        """Refine features using one global pool of ensemble perturbations.

        Unlike :meth:`stochastic_feature_selection`, this method does not rank
        candidates within each model. It constructs every feasible
        ``(model, operation, perturbation group)`` candidate and estimates its
        effect on one aggregate out-of-fold ensemble prediction. One candidate
        is sampled from this global ranking and fully tuned per iteration.

        Candidate effects use the fitted model's decision perturbation, so
        scoring the complete pool does not require refitting every candidate.
        Acceptance uses the actual out-of-fold ensemble score after tuning the
        selected proposal. The best accepted state is restored on return.
        When ``preserve_feature_count`` is true, the global pool instead
        contains matched remove/add swaps within each model. Only perturbation
        groups of equal size are paired, so every model retains its initial
        feature count throughout the search.

        Diversity can be rewarded through mean pairwise feature-set Jaccard
        distance and mean pairwise out-of-fold decision-value decorrelation.
        ``performance_tolerance`` limits the score loss relative to the best
        score seen, while ``max_feature_similarity`` can reject candidates
        whose pairwise feature Jaccard similarity is too high.
        """
        if (isinstance(n_iterations, (bool, np.bool_)) or
                not isinstance(n_iterations, (int, np.integer))):
            raise TypeError("n_iterations must be an integer")
        if n_iterations < 1:
            raise ValueError("n_iterations must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if not 0 < cooling_rate <= 1:
            raise ValueError("cooling_rate must be in (0, 1]")
        if not 0 <= add_probability <= 1:
            raise ValueError("add_probability must be between 0 and 1")
        if not isinstance(preserve_feature_count, (bool, np.bool_)):
            raise TypeError("preserve_feature_count must be boolean")
        for name, value in (("feature_diversity_weight",
                             feature_diversity_weight),
                            ("prediction_diversity_weight",
                             prediction_diversity_weight)):
            if (isinstance(value, (bool, np.bool_)) or
                    not isinstance(value, (int, float, np.integer, np.floating))):
                raise TypeError(f"{name} must be numeric")
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if performance_tolerance is not None:
            if (not np.isfinite(performance_tolerance) or
                    performance_tolerance < 0):
                raise ValueError(
                    "performance_tolerance must be finite and non-negative")
            performance_tolerance = float(performance_tolerance)
        if max_feature_similarity is not None:
            if (not np.isfinite(max_feature_similarity) or
                    not 0 <= max_feature_similarity <= 1):
                raise ValueError("max_feature_similarity must be in [0, 1]")
            max_feature_similarity = float(max_feature_similarity)
        feature_diversity_weight = float(feature_diversity_weight)
        prediction_diversity_weight = float(prediction_diversity_weight)
        if convergence_patience is not None:
            if (isinstance(convergence_patience, (bool, np.bool_)) or
                    not isinstance(convergence_patience, (int, np.integer))):
                raise TypeError("convergence_patience must be an integer or None")
            if convergence_patience < 1:
                raise ValueError("convergence_patience must be at least 1")
        if (not np.isfinite(convergence_min_delta) or
                convergence_min_delta < 0):
            raise ValueError(
                "convergence_min_delta must be finite and non-negative")
        parameter_grid = list(parameter_grid)
        if not parameter_grid:
            raise ValueError("parameter_grid must contain at least one candidate")
        if not hasattr(self, "parameters_"):
            raise RuntimeError("models must be fitted before ensemble refinement")
        if not self.separate_feature_sets:
            raise ValueError(
                "ensemble pooled refinement requires separate_feature_sets=True")

        rng = np.random.default_rng(random_seed)
        score_weight = float(getattr(
            getattr(self.score, "__self__", None), "weight", 0.5))

        def score_predictions(indices, predictions):
            y_true = self.cv.y[indices]
            if self._is_one_class():
                labels = np.unique(y_true)
                if labels.size == 1:
                    return float(np.mean(np.asarray(predictions) >= 0))
                predicted_labels = np.where(np.asarray(predictions) >= 0, 1, -1)
                f1 = f1_score(y_true, predicted_labels, pos_label=1)
                auc = roc_auc_score(y_true, predictions)
                return float(score_weight * auc + (1-score_weight) * f1)
            if isinstance(self.SVM, SVR):
                if len(np.unique(predictions)) <= 2:
                    pearson = r2 = 0.00001
                else:
                    pearson = pearsonr(y_true, predictions).statistic ** 2
                    r2 = r2_score(y_true, predictions)
                return float(score_weight * pearson +
                             (1 - score_weight) * max(0.00001, r2))

            classes = np.asarray(self.models[0].classes_)
            if classes.size != 2:
                raise ValueError(
                    "ensemble pooled refinement requires binary classification")
            predicted_labels = classes[(predictions > 0).astype(int)]
            f1 = f1_score(y_true, predicted_labels, pos_label=classes[1])
            auc = roc_auc_score(y_true, predictions)
            return float(score_weight * auc + (1 - score_weight) * f1)

        def out_of_fold_state():
            sums = np.zeros(len(self.cv.y), dtype=float)
            counts = np.zeros(len(self.cv.y), dtype=int)
            outputs = []
            for model_index in range(self.num_models):
                indices = np.asarray(self.cv.test[model_index], dtype=int)
                if isinstance(self.SVM, SVR):
                    kernel = self._inference_kernel(self.cv.X[indices], model_index)
                    values = self.models[model_index].predict(kernel)
                else:
                    values = self.decision_function(
                        self.cv.X[indices], model_index=model_index)
                outputs.append(np.asarray(values, dtype=float))
                sums[indices] += values
                counts[indices] += 1
            covered = np.flatnonzero(counts > 0)
            if covered.size == 0:
                raise ValueError(
                    "out-of-fold ensemble scoring requires non-empty CV test sets")
            aggregate = sums[covered] / counts[covered]
            return sums, counts, covered, aggregate, outputs

        def feature_diversity(feature_sets):
            distances = []
            similarities = []
            normalized = [set(np.asarray(features).tolist())
                          for features in feature_sets]
            for first in range(len(normalized)):
                for second in range(first + 1, len(normalized)):
                    union = normalized[first] | normalized[second]
                    similarity = (len(normalized[first] & normalized[second]) /
                                  len(union) if union else 1.0)
                    similarities.append(similarity)
                    distances.append(1.0 - similarity)
            return (float(np.mean(distances)) if distances else 0.0,
                    float(np.max(similarities)) if similarities else 0.0)

        def prediction_diversity(outputs):
            distances = []
            for first in range(self.num_models):
                first_indices = np.asarray(self.cv.test[first], dtype=int)
                for second in range(first + 1, self.num_models):
                    second_indices = np.asarray(self.cv.test[second], dtype=int)
                    _, first_positions, second_positions = np.intersect1d(
                        first_indices, second_indices, return_indices=True)
                    if len(first_positions) < 2:
                        continue
                    first_values = outputs[first][first_positions]
                    second_values = outputs[second][second_positions]
                    if isinstance(self.SVM, SVR):
                        overlap_y = self.cv.y[first_indices[first_positions]]
                        first_values = overlap_y - first_values
                        second_values = overlap_y - second_values
                    first_std = np.std(first_values)
                    second_std = np.std(second_values)
                    if first_std == 0 or second_std == 0:
                        correlation = (1.0 if np.allclose(
                            first_values, second_values) else 0.0)
                    else:
                        correlation = np.corrcoef(
                            first_values, second_values)[0, 1]
                    distances.append(1.0 - abs(float(correlation)))
            return float(np.mean(distances)) if distances else 0.0

        def combined_objective(score, feature_distance,
                               prediction_distance):
            return float(score +
                         feature_diversity_weight * feature_distance +
                         prediction_diversity_weight * prediction_distance)

        def candidate_feature_sets(candidate):
            proposed = [np.asarray(features, dtype=int).copy()
                        for features in self.features]
            model_index = candidate.model_index
            if candidate.operation == "swap":
                proposed[model_index] = proposed[model_index][
                    ~np.isin(proposed[model_index], candidate.features_removed)]
                proposed[model_index] = np.concatenate((
                    proposed[model_index], candidate.features_added))
            elif candidate.operation == "add":
                proposed[model_index] = np.concatenate((
                    proposed[model_index], candidate.features_changed))
            else:
                proposed[model_index] = proposed[model_index][
                    ~np.isin(proposed[model_index], candidate.features_changed)]
            return proposed

        def snapshot():
            return copy.deepcopy((
                self.models, self.kernel_matrix_, self.parameters_,
                self.performance_, self.features,
                getattr(self, "decision_value_cutoff_", 0.0),
                self._kernel_configuration_, self.removed_features_))

        def restore(state):
            (self.models, self.kernel_matrix_, self.parameters_,
             self.performance_, self.features, self.decision_value_cutoff_,
             self._kernel_configuration_,
             self.removed_features_) = copy.deepcopy(state)
            self._update_unified_feature_attributes()

        def add_diversity(candidate, proposed_outputs):
            candidate_features = candidate_feature_sets(candidate)
            feature_distance, maximum_similarity = feature_diversity(
                candidate_features)
            prediction_distance = prediction_diversity(proposed_outputs)
            candidate.estimated_feature_diversity = feature_distance
            candidate.estimated_prediction_diversity = prediction_distance
            candidate.max_feature_similarity = maximum_similarity
            candidate.estimated_objective = combined_objective(
                candidate.estimated_score, feature_distance,
                prediction_distance)
            if (max_feature_similarity is not None and
                    maximum_similarity > max_feature_similarity and
                    maximum_similarity >= current_max_similarity):
                return False
            if (performance_tolerance is not None and
                    candidate.estimated_score < peak_score - performance_tolerance):
                return False
            return True

        def candidate_pool(sums, counts, covered, outputs):
            candidates = []
            previous_direction = getattr(self, "_selection_direction_", None)
            try:
                if preserve_feature_count:
                    for model_index in range(self.num_models):
                        active = self._active_perturbation_sets(model_index)
                        inactive = self._inactive_perturbation_sets(model_index)
                        if len(active) <= 1 or not inactive:
                            continue
                        indices = np.asarray(
                            self.cv.test[model_index], dtype=int)
                        self._selection_direction_ = "backward"
                        remove_effects = self.decision_perturbation_(
                            model_index, self.cv.X[indices])
                        self._selection_direction_ = "forward"
                        add_effects = self.decision_perturbation_(
                            model_index, self.cv.X[indices])
                        for remove_index, remove_group in enumerate(active):
                            for add_index, add_group in enumerate(inactive):
                                if len(remove_group) != len(add_group):
                                    continue
                                proposed_sums = sums.copy()
                                proposed_sums[indices] -= (
                                    remove_effects[:, remove_index] +
                                    add_effects[:, add_index])
                                proposed = proposed_sums[covered] / counts[covered]
                                candidate = dotdict({
                                    "model_index": model_index,
                                    "operation": "swap",
                                    "features_removed": remove_group.copy(),
                                    "features_added": add_group.copy(),
                                    "features_changed": np.concatenate((
                                        remove_group, add_group)),
                                    "estimated_score": score_predictions(
                                        covered, proposed),
                                })
                                proposed_outputs = list(outputs)
                                proposed_outputs[model_index] = (
                                    outputs[model_index] -
                                    remove_effects[:, remove_index] -
                                    add_effects[:, add_index])
                                if add_diversity(candidate, proposed_outputs):
                                    candidates.append(candidate)
                    return candidates
                for operation in ("add", "remove"):
                    self._selection_direction_ = (
                        "forward" if operation == "add" else "backward")
                    for model_index in range(self.num_models):
                        groups = (self._inactive_perturbation_sets(model_index)
                                  if operation == "add" else
                                  self._active_perturbation_sets(model_index))
                        if operation == "remove" and len(groups) <= 1:
                            continue
                        if not groups:
                            continue
                        indices = np.asarray(self.cv.test[model_index], dtype=int)
                        perturbations = self.decision_perturbation_(
                            model_index, self.cv.X[indices])
                        for group_index, group in enumerate(groups):
                            proposed_sums = sums.copy()
                            # decision_perturbation_ is current minus the
                            # estimated perturbed decision for both directions.
                            proposed_sums[indices] -= perturbations[:, group_index]
                            proposed = proposed_sums[covered] / counts[covered]
                            candidate = dotdict({
                                "model_index": (model_index
                                                if self.separate_feature_sets
                                                else None),
                                "operation": operation,
                                "features_changed": group.copy(),
                                "estimated_score": score_predictions(
                                    covered, proposed),
                            })
                            proposed_outputs = list(outputs)
                            proposed_outputs[model_index] = (
                                outputs[model_index] -
                                perturbations[:, group_index])
                            if add_diversity(candidate, proposed_outputs):
                                candidates.append(candidate)
            finally:
                if previous_direction is None:
                    self.__dict__.pop("_selection_direction_", None)
                else:
                    self._selection_direction_ = previous_direction
            return candidates

        sums, counts, covered, aggregate, outputs = out_of_fold_state()
        current_score = score_predictions(covered, aggregate)
        current_feature_diversity, current_max_similarity = feature_diversity(
            self.features)
        current_prediction_diversity = prediction_diversity(outputs)
        current_objective = combined_objective(
            current_score, current_feature_diversity,
            current_prediction_diversity)
        current_state = snapshot()
        best_score = current_score
        peak_score = current_score
        best_objective = current_objective
        best_feature_diversity = current_feature_diversity
        best_prediction_diversity = current_prediction_diversity
        best_state = snapshot()
        convergence_objective = current_objective
        stale_iterations = 0
        current_temperature = float(temperature)
        history = []
        converged = False
        stop_reason = "max_iterations"

        for iteration in range(int(n_iterations)):
            candidates = candidate_pool(sums, counts, covered, outputs)
            if not candidates:
                stop_reason = "no_feasible_moves"
                break
            # Preserve operation probability without ever producing an empty
            # pool at a feature-space boundary.
            if preserve_feature_count:
                operation_pool = candidates
            else:
                choose_add = rng.random() < add_probability
                operation_pool = [candidate for candidate in candidates
                                  if candidate.operation == (
                                      "add" if choose_add else "remove")]
                if not operation_pool:
                    operation_pool = candidates
            ranks = rank_items(np.asarray([
                candidate.estimated_objective for candidate in operation_pool]))
            weights = ranks.astype(float) + 1.0
            selected = operation_pool[int(rng.choice(
                len(operation_pool), p=weights / np.sum(weights)))]

            if selected.operation == "swap":
                self._remove_features(
                    selected.features_removed,
                    model_index=selected.model_index, update_kernel=False)
                self._add_features(
                    selected.features_added,
                    model_index=selected.model_index, update_kernel=False)
            elif selected.operation == "add":
                self._add_features(
                    selected.features_changed,
                    model_index=selected.model_index, update_kernel=False)
            else:
                self._remove_features(
                    selected.features_changed,
                    model_index=selected.model_index, update_kernel=False)
            self._tune_member_models(parameter_grid)
            proposed_sums, proposed_counts, proposed_covered, \
                proposed_aggregate, proposed_outputs = out_of_fold_state()
            proposed_score = score_predictions(
                proposed_covered, proposed_aggregate)
            proposed_feature_diversity, proposed_max_similarity = \
                feature_diversity(self.features)
            proposed_prediction_diversity = prediction_diversity(
                proposed_outputs)
            proposed_objective = combined_objective(
                proposed_score, proposed_feature_diversity,
                proposed_prediction_diversity)
            objective_change = proposed_objective - current_objective
            meets_performance = (
                performance_tolerance is None or
                proposed_score >= peak_score - performance_tolerance)
            meets_similarity = (
                max_feature_similarity is None or
                proposed_max_similarity <= max_feature_similarity or
                proposed_max_similarity < current_max_similarity)
            accepted = bool(
                meets_performance and meets_similarity and
                (objective_change >= 0 or rng.random() <
                 np.exp(objective_change / current_temperature)))
            if accepted:
                current_score = proposed_score
                current_objective = proposed_objective
                current_feature_diversity = proposed_feature_diversity
                current_prediction_diversity = proposed_prediction_diversity
                current_max_similarity = proposed_max_similarity
                current_state = snapshot()
                sums, counts, covered, outputs = (
                    proposed_sums, proposed_counts, proposed_covered,
                    proposed_outputs)
                peak_score = max(peak_score, proposed_score)
                best_outside_tolerance = (
                    performance_tolerance is not None and
                    best_score < peak_score - performance_tolerance)
                if (proposed_objective > best_objective or
                        best_outside_tolerance):
                    best_score = proposed_score
                    best_objective = proposed_objective
                    best_feature_diversity = proposed_feature_diversity
                    best_prediction_diversity = proposed_prediction_diversity
                    best_state = snapshot()
                if (proposed_objective >
                        convergence_objective + convergence_min_delta):
                    convergence_objective = proposed_objective
                    stale_iterations = 0
                else:
                    stale_iterations += 1
            else:
                restore(current_state)
                stale_iterations += 1

            history.append(dotdict({
                "iteration": iteration,
                "model_index": selected.model_index,
                "operation": selected.operation,
                "features_changed": selected.features_changed.copy(),
                "features_removed": copy.deepcopy(getattr(
                    selected, "features_removed", None)),
                "features_added": copy.deepcopy(getattr(
                    selected, "features_added", None)),
                "estimated_score": selected.estimated_score,
                "estimated_objective": selected.estimated_objective,
                "estimated_feature_diversity":
                    selected.estimated_feature_diversity,
                "estimated_prediction_diversity":
                    selected.estimated_prediction_diversity,
                "score": proposed_score,
                "objective": proposed_objective,
                "feature_diversity": proposed_feature_diversity,
                "prediction_diversity": proposed_prediction_diversity,
                "max_feature_similarity": proposed_max_similarity,
                "meets_performance_tolerance": meets_performance,
                "meets_feature_similarity": meets_similarity,
                "accepted": accepted,
                "temperature": current_temperature,
                "pool_size": len(candidates),
                "iterations_without_improvement": stale_iterations,
                "features": copy.deepcopy(self.features),
            }))
            current_temperature *= cooling_rate
            if (convergence_patience is not None and
                    stale_iterations >= convergence_patience):
                converged = True
                stop_reason = "converged"
                break

        restore(best_state)
        self.ensemble_stochastic_performance_ = history
        self.ensemble_stochastic_best_score_ = best_score
        self.ensemble_stochastic_best_objective_ = best_objective
        self.ensemble_stochastic_best_feature_diversity_ = \
            best_feature_diversity
        self.ensemble_stochastic_best_prediction_diversity_ = \
            best_prediction_diversity
        self.ensemble_stochastic_peak_score_ = peak_score
        self.ensemble_stochastic_converged_ = converged
        self.ensemble_stochastic_iterations_ = len(history)
        self.ensemble_stochastic_stop_reason_ = stop_reason
        self.fit_unified_model(parameter_grid)
        return self


    def set_num_features(self, num_features, parameter_grid):
        """Use the top-ranked features and retune the ensemble.

        This method is intended for use after a feature-selection method has
        populated ``sorted_features``. For ensembles with separate feature
        sets, each model uses the top features from its own ranking.

        Parameters
        ----------
        num_features : int
            Number of highest-ranked features to retain.
        parameter_grid : iterable
            Parameter candidates accepted by :meth:`tune_models`.
        """
        if not hasattr(self, "sorted_features"):
            raise RuntimeError(
                "feature selection must be run before setting the number of features")
        if (isinstance(num_features, (bool, np.bool_)) or
                not isinstance(num_features, (int, np.integer))):
            raise TypeError("num_features must be an integer")
        if num_features < 1:
            raise ValueError("num_features must be at least 1")

        rankings = (self.sorted_features if self.separate_feature_sets
                    else [self.sorted_features])
        expected_rankings = self.num_models if self.separate_feature_sets else 1
        if len(rankings) != expected_rankings:
            raise RuntimeError("feature rankings do not match the ensemble")
        if any(num_features > len(ranking) for ranking in rankings):
            raise ValueError(
                "num_features cannot exceed the number of ranked features")

        if self.separate_feature_sets:
            for model_index, ranking in enumerate(rankings):
                self._set_features(
                    np.asarray(ranking)[:num_features],
                    model_index=model_index,
                    update_kernel=False)
        else:
            self._set_features(
                np.asarray(rankings[0])[:num_features],
                update_kernel=False)

        self.tune_models(parameter_grid)
            
    
    def greedy_backward_selection(self, parameter_grid, 
                                  reduction_factor = 0.1, 
                                  feature_ranker = combined_rank().compute, 
                                  set_for_rank = "train",
                                  tune_models_each_step = True,
                                  post_find_knee = True):
        """Rank and remove feature sets using greedy backward selection.

        When ``tune_models_each_step`` is false, tuning is performed only for
        the initial full-feature model. Its selected parameters are retained,
        while gamma, when present, is scaled at each later step as
        ``initial_gamma * initial_feature_count / current_feature_count``.
        Kernels without a gamma parameter retain their tuned parameters. Once
        the best feature subset has been selected, that subset is retuned with
        the full parameter grid so the returned model is not left at the
        search-time scaled parameters. When ``post_find_knee`` is true, the
        completed performance curve is used to select the knee feature count
        before this final retuning pass. If the curve has no detectable knee,
        the best-scoring subset is retained.
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
                self._tune_member_models(parameter_grid)
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
        self._update_unified_feature_attributes()

        self.feature_performance_ = feature_performance

        self.models = best_models
        self.parameters_ = best_parameters
        self.performance_ = best_performance
        self.features = best_features
        self.kernel_matrix_ = best_kernel_matrix
        self._update_unified_feature_attributes()
        if post_find_knee:
            try:
                knee_num_features = self.find_knee()
            except ValueError as error:
                warnings.warn(
                    f"Could not select a post-search knee ({error}); "
                    "retaining the best-scoring feature set.",
                    RuntimeWarning,
                    stacklevel=2)
                if not tune_models_each_step:
                    self._tune_member_models(parameter_grid)
            else:
                self.set_num_features(knee_num_features, parameter_grid)
        elif not tune_models_each_step:
            self.tune_models(parameter_grid)
        if self.unified_model_ is None:
            self.fit_unified_model(parameter_grid)


    def greedy_forward_selection(self, parameter_grid,
                                 addition_factor=0.1,
                                 feature_ranker=combined_rank().compute,
                                 set_for_rank="train",
                                 tune_models_each_step=True,
                                 max_features=None,
                                 post_find_knee=True):
        """Rank and add feature sets using greedy forward selection.

        Every perturbation set is fitted by itself in the first round and the
        best singleton is retained.  Later rounds use the same perturbation
        ranker as backward selection, but perturb inactive sets by adding them;
        consequently forward decision perturbations have the opposite sign.
        When ``max_features`` is supplied, the greedy search stops at that
        many active feature columns. Unselected features tie for the final
        rank, and a full-feature model is still evaluated for comparison but
        is not eligible to replace the best capped model. When
        ``tune_models_each_step`` is false, search-time gamma scaling is used
        after the initial tuning pass and the selected best subset is retuned
        once with the full parameter grid before returning. When
        ``post_find_knee`` is true, the completed performance curve is used to
        select the knee feature count before that final retuning pass. If the
        curve has no detectable knee, the best-scoring subset is retained.
        ``addition_factor`` controls the fraction of currently inactive
        perturbation sets added per iteration. A value of zero adds exactly
        one set at a time.
        """
        if not isinstance(addition_factor, (int, float, np.integer, np.floating)):
            raise TypeError("addition_factor must be numeric")
        if addition_factor < 0:
            raise ValueError("addition_factor must be nonnegative")

        def number_to_add(num_inactive):
            if addition_factor == 0:
                return 1
            return max(1, min(
                num_inactive,
                int(np.floor(num_inactive * addition_factor))))

        candidates = [np.asarray(group, dtype=int)
                      for group in self.perturbation_sets]
        if not candidates:
            raise ValueError("forward selection requires at least one feature set")

        medoids = set(getattr(
            self.cv, "feature_medoids_", np.arange(self.cv.X.shape[1])).tolist())
        candidates = [candidate for candidate in candidates
                      if medoids.intersection(candidate.tolist())]
        self.singleton_candidates_ = copy.deepcopy(candidates)

        total_features = self.cv.X.shape[1]
        if max_features is None:
            max_features = total_features
        elif not isinstance(max_features, (int, np.integer)):
            raise TypeError("max_features must be an integer or None")
        elif max_features < 1 or max_features > total_features:
            raise ValueError(
                f"max_features must be between 1 and {total_features}")

        eligible_candidates = [
            candidate for candidate in candidates
            if len(candidate) <= max_features]
        if not eligible_candidates:
            raise ValueError(
                "max_features is smaller than every perturbation set")

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
            for candidate_index, candidate in enumerate(eligible_candidates):
                if self.separate_feature_sets:
                    for model_index in range(self.num_models):
                        self._set_features(candidate, model_index,
                                           update_kernel=False)
                else:
                    self._set_features(candidate, update_kernel=False)
                self._tune_member_models(parameter_grid)
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

            def fit_current_feature_set():
                if tune_models_each_step:
                    self._tune_member_models(parameter_grid)
                    return

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
                current_count = (np.mean([len(features) for features in self.features])
                                 if self.separate_feature_sets else len(self.features))
                if not inactive or current_count >= max_features:
                    break

                if self.separate_feature_sets:
                    added_any = False
                    for model_index in range(self.num_models):
                        model_inactive = self._inactive_perturbation_sets(model_index)
                        ranks = feature_ranker(self, model_index, set_for_rank)
                        n_to_add = number_to_add(len(model_inactive))
                        ranked = np.argsort(ranks)[::-1]
                        chosen = []
                        feature_count = len(self.features[model_index])
                        for index in ranked:
                            if len(chosen) >= n_to_add:
                                break
                            if feature_count + len(model_inactive[index]) <= max_features:
                                chosen.append(index)
                                feature_count += len(model_inactive[index])
                        if not chosen:
                            continue
                        additions = np.concatenate([
                            model_inactive[index] for index in chosen])
                        selection_order[model_index].extend(additions.tolist())
                        self._add_features(additions,
                            model_index, update_kernel=False)
                        added_any = True
                    if not added_any:
                        break
                else:
                    rank_total = np.zeros(len(inactive))
                    for model_index in range(self.num_models):
                        rank_total += feature_ranker(self, model_index, set_for_rank)
                    consensus = rank_items(rank_total)
                    n_to_add = number_to_add(len(inactive))
                    ranked = np.argsort(consensus)[::-1]
                    chosen = []
                    feature_count = len(self.features)
                    for index in ranked:
                        if len(chosen) >= n_to_add:
                            break
                        if feature_count + len(inactive[index]) <= max_features:
                            chosen.append(index)
                            feature_count += len(inactive[index])
                    if not chosen:
                        break
                    additions = np.concatenate([
                        inactive[index] for index in chosen])
                    selection_order.extend(additions.tolist())
                    self._add_features(additions, update_kernel=False)

                fit_current_feature_set()

            # Preserve a full-feature performance endpoint even when the
            # greedy search is capped. It is intentionally not passed to
            # save_state, so it cannot replace the best capped model.
            if max_features < total_features:
                if self.separate_feature_sets:
                    for model_index in range(self.num_models):
                        self._set_features(np.arange(total_features), model_index,
                                           update_kernel=False)
                else:
                    self._set_features(np.arange(total_features),
                                       update_kernel=False)
                fit_current_feature_set()
                row = mean_row()
                row["num_features"] = total_features
                row["mean_nSV"] = np.sum([
                    np.sum(model.n_support_) for model in self.models
                ]) / self.num_models
                print(f"Number of Features: {row['num_features']:.0f}, "
                      f"Score: {row['score']:.3f}")
                feature_performance[result] = copy.deepcopy(row)

            self.singleton_performance_ = singleton_performance
            self.feature_performance_ = feature_performance
            self.models, self.kernel_matrix_, self.parameters_, \
                self.performance_, self.features = best_state
            self._kernel_configuration_ = self._kernel_configuration(
                self.parameters_)
            if self.separate_feature_sets:
                self.sorted_features = []
                self.feature_rank = []
                for model_index, selected in enumerate(selection_order):
                    unselected = [feature for feature in range(total_features)
                                  if feature not in selected]
                    ordered = np.asarray(selected + unselected)
                    ranks = np.empty(total_features, dtype=int)
                    for rank, feature in enumerate(selected):
                        ranks[feature] = rank
                    ranks[unselected] = len(selected)
                    self.sorted_features.append(ordered)
                    self.feature_rank.append(ranks)
            else:
                unselected = [feature for feature in range(total_features)
                              if feature not in selection_order]
                self.sorted_features = np.asarray(selection_order + unselected)
                self.feature_rank = np.empty(total_features, dtype=int)
                for rank, feature in enumerate(selection_order):
                    self.feature_rank[feature] = rank
                self.feature_rank[unselected] = len(selection_order)
            self._update_unified_feature_attributes()
            if post_find_knee:
                try:
                    knee_num_features = self.find_knee()
                except ValueError as error:
                    warnings.warn(
                        f"Could not select a post-search knee ({error}); "
                        "retaining the best-scoring feature set.",
                        RuntimeWarning,
                        stacklevel=2)
                    if not tune_models_each_step:
                        self.tune_models(parameter_grid)
                else:
                    self.set_num_features(knee_num_features, parameter_grid)
            elif not tune_models_each_step:
                self.tune_models(parameter_grid)
            if self.unified_model_ is None:
                self.fit_unified_model(parameter_grid)
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
        dual_coef = self.models[model_index].dual_coef_[0, :]
        const = -0.5*(np.dot(dual_coef, dual_coef.transpose()))
        
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
            
            if self._is_one_class():
                # Frozen-coefficient change in 1/2 ||w||^2.  Its magnitude is
                # used because non-additive kernels need not give the change
                # a consistent sign when a feature group is removed/added.
                criteria[z] = abs(
                    0.5 * dual_coef @ (K-Kp) @ dual_coef)
            else:
                # Preserve the established SVC/SVR criterion exactly.
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

        When ``model_index`` is omitted, the result is the mean attribution
        across all models.  For separate feature sets, its columns correspond
        to the sorted union of the models' feature indices; a model contributes
        zero for every feature it does not use.

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

        if model_index is None:
            model_indices = [i for i in range(self.num_models)]
        else:
            model_indices = [model_index]

        if self.separate_feature_sets:
            if model_index is None:
                self._update_unified_feature_attributes()
                features = self.unified_features
            else:
                features = np.asarray(self.features[model_index], dtype=int)
        else:
            features = np.asarray(self.features, dtype=int)

        feature_positions = {
            feature: position for position, feature in enumerate(features)
        }
        integrated_gradient = np.zeros([len(X), len(features)])
        for m in model_indices:
            model_features = (np.asarray(self.features[m], dtype=int)
                              if self.separate_feature_sets else features)
            output_positions = [feature_positions[feature]
                                for feature in model_features]
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
                # Standard integrated gradients attribute the change from the
                # reference output, not the absolute output.  Consequently no
                # reference prediction is distributed across features, and an
                # SVR attribution is not divided by its feature count.  Up to
                # numerical integration error, summing these contributions
                # yields f(x) - f(reference) for both regression and the SVC
                # decision function.
                model_gradient = x_diff[model_features] * np.asarray([
                    np.trapz(gradient_steps[:, n], x=path_fraction[:, 0])
                    for n in range(gradient_steps.shape[1])
                ])
                integrated_gradient[i, output_positions] += model_gradient

        return integrated_gradient/len(model_indices)

    def explain_integrated_gradients(self, X, feature_names=None, target=None,
                                     model_index=None, num_steps=20,
                                     reference_point=None, ref_point=None):
        """Return integrated gradients together with plotting metadata."""
        X = np.asarray(X)
        values = self.integrated_gradient(
            X, model_index=model_index, num_steps=num_steps,
            reference_point=reference_point, ref_point=ref_point)
        if self.separate_feature_sets:
            if model_index is None:
                self._update_unified_feature_attributes()
                features = np.asarray(self.unified_features, dtype=int)
            else:
                features = np.asarray(self.features[model_index], dtype=int)
        else:
            features = np.asarray(self.features, dtype=int)
        if feature_names is None:
            names = tuple(f"feature_{feature}" for feature in features)
        else:
            feature_names = list(feature_names)
            if len(feature_names) == X.shape[1]:
                names = tuple(str(feature_names[feature]) for feature in features)
            elif len(feature_names) == len(features):
                names = tuple(map(str, feature_names))
            else:
                raise ValueError("feature_names must describe all input or selected features")
        supplied_reference = reference_point if reference_point is not None else ref_point
        references = None
        if supplied_reference is not None:
            references = np.broadcast_to(np.asarray(supplied_reference), X.shape).copy()[:, features]
        model_indices = (tuple(range(self.num_models)) if model_index is None
                         else (int(model_index),))
        return IntegratedGradientsResult(
            values=values, inputs=X[:, features], feature_indices=features,
            feature_names=names, reference_points=references,
            model_indices=model_indices, num_steps=num_steps, target=target)

    
    def plot_performance(self,metric = 'score'):
        x = [self.feature_performance_[key]['num_features'] for key in self.feature_performance_.keys()]
        y = [self.feature_performance_[key][metric] for key in self.feature_performance_.keys()]
        
        plt.plot(x,y)
        plt.xlabel('# of features') 
        plt.ylabel(metric) 


    def find_knee(self, metric='score'):
        """Return the feature count at the knee of a performance curve.

        The curve is sorted by feature count and normalized to the unit
        square. The knee is the interior point with the greatest vertical
        distance above the diagonal, corresponding to the point after which
        adding features produces diminishing gains in ``metric``.

        Parameters
        ----------
        metric : str, default="score"
            Higher-is-better performance value stored in each row of
            ``feature_performance_``.

        Returns
        -------
        int or float
            Number of features at the knee. The value is also stored in
            ``knee_num_features_``.
        """
        if not hasattr(self, "feature_performance_"):
            raise RuntimeError(
                "feature selection must be run before finding a knee")

        try:
            points = np.asarray([
                (row["num_features"], row[metric])
                for row in self.feature_performance_.values()
            ], dtype=float)
        except KeyError as error:
            raise KeyError(
                f"feature performance does not contain {error.args[0]!r}") from error

        if points.ndim != 2 or points.shape[0] < 3:
            raise ValueError("at least three performance points are required")
        if not np.all(np.isfinite(points)):
            raise ValueError("feature counts and performance values must be finite")

        # Keep the strongest result when multiple selection steps have the
        # same feature count, then make the result independent of traversal
        # direction (forward or backward selection).
        feature_counts = np.unique(points[:, 0])
        performance = np.asarray([
            np.max(points[points[:, 0] == count, 1])
            for count in feature_counts
        ])
        if len(feature_counts) < 3:
            raise ValueError("at least three distinct feature counts are required")

        feature_range = np.ptp(feature_counts)
        performance_range = np.ptp(performance)
        if feature_range == 0 or performance_range == 0:
            raise ValueError("a knee cannot be found in a flat curve")

        normalized_features = (
            (feature_counts - feature_counts[0]) / feature_range)
        normalized_performance = (
            (performance - np.min(performance)) / performance_range)
        distance = normalized_performance - normalized_features
        interior_distance = distance[1:-1]
        knee_offset = int(np.argmax(interior_distance))
        if interior_distance[knee_offset] <= np.finfo(float).eps:
            raise ValueError("the performance curve does not contain a knee")

        knee = feature_counts[knee_offset + 1]
        if knee.is_integer():
            knee = int(knee)
        self.knee_num_features_ = knee
        return knee


    def predict(self, X, model_index=None, use_voting=False,
                prediction_mode="unified"):
        """Predict with the unified model by default.

        Pass ``prediction_mode='set'`` to average member outputs as in older
        releases. Supplying ``model_index`` continues to select one member;
        ``use_voting=True`` likewise implies set-based classification.
        """
        if prediction_mode not in {"unified", "set"}:
            raise ValueError("prediction_mode must be 'unified' or 'set'")
        if model_index is None and not use_voting and prediction_mode == "unified":
            # Serialized models from releases before unified prediction do
            # not contain a final model and retain their original set output.
            if self.unified_model_ is not None:
                kernel_matrix = self._unified_inference_kernel(X)
                return self.unified_model_.predict(kernel_matrix)
            prediction_mode = "set"

        if self._is_one_class():
            if model_index is None:
                model_indices = range(self.num_models)
            else:
                model_indices = [model_index]
            if use_voting:
                votes = np.zeros(len(X), dtype=float)
                for m in model_indices:
                    votes += self.models[m].predict(
                        self._inference_kernel(X, m))
                predictions = np.where(votes >= 0, 1, -1)
            else:
                decision_values = self.decision_function(
                    X, model_index, prediction_mode="set")
                predictions = np.where(decision_values >= 0, 1, -1)

        elif isinstance(self.SVM, SVR):
            if model_index == None:
                model_indices = [i for i in range(self.num_models)]
            else:
                model_indices = [model_index]
    
            predictions = 0
            for m in model_indices:
                kernel_matrix = self._inference_kernel(X, m)
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
                    kernel_matrix = self._inference_kernel(X, m)
                    model_predictions = self.models[m].predict(kernel_matrix)
                    prediction_counts += (model_predictions == positive_class) + 0
                
                predictions = self.models[0].classes_[(prediction_counts/len(model_indices) < 0.5) + 0] 
                
            else:
                decision_values = self.decision_function(
                    X, model_index, prediction_mode="set")
                cutoff = (getattr(self, "decision_value_cutoff_", 0.0)
                          if model_index is None else 0.0)
                predictions = self.models[0].classes_[
                    (decision_values > cutoff) + 0]

        return predictions
        

    def _set_decision_function(self, X, model_index=None):
        if model_index == None:
            model_indices = [i for i in range(self.num_models)]
        else:
            model_indices = [model_index]

        decision_values = 0
        for m in model_indices:
            kernel_matrix = self._inference_kernel(X, m)
            decision_values += self.models[m].decision_function(kernel_matrix)

        return decision_values/len(model_indices)


    def decision_function(self, X, model_index=None, prediction_mode="unified"):
        """Return unified decision values, or member-set values on request."""
        if prediction_mode not in {"unified", "set"}:
            raise ValueError("prediction_mode must be 'unified' or 'set'")
        if model_index is None and prediction_mode == "unified":
            if self.unified_model_ is not None:
                return self.unified_model_.decision_function(
                    self._unified_inference_kernel(X))
        return self._set_decision_function(X, model_index)


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
        
