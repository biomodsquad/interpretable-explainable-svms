import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   

from scipy.optimize import minimize
from scipy.sparse import dok_matrix

import copy
from collections import Counter

from sklearn.svm import SVR

import random

from mistic.utility import combined_rank, kernelWrapper, score_svr, score_svc, dotdict, svc_dec2, rank_items

class svmSet():
    """
    A class used to represent a set of Support Vector Machines (SVMs) for feature selection.

    Attributes
    ----------
    SVM : object
        An instance of an SVM model.
    cv : object
        Cross-validation set.
    score : function
        Method to score the SVM.
    kernel : function, optional
        Kernel function to be used by the SVM (default is kernelWrapper().compute).
    separate_feature_sets : bool, optional
        Flag to indicate if separate feature sets should be used (default is False).
    separate_parameters : bool, optional
        Flag to indicate if separate parameters should be used (default is False).

    Methods
    -------
    __init__(self, SVM, cvSet, score_method, kernel=kernelWrapper().compute, separate_feature_sets=False, separate_parameters=False)
        Initializes the svmSet with the given parameters.
    """   
    def __init__(self, SVM, cvSet, score_method, 
                 kernel = kernelWrapper().compute,
                 separate_feature_sets = False,
                 separate_parameters = False,
                 sparse_kernel_matrix = False):
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

        self.sparse_kernel_matrix = sparse_kernel_matrix
            
        self.kernel = kernel
        self._reset_kernel_matrix()
    
        self.score = score_method

        self.models = []
        self.X_ind = []
        for i in range(self.num_models):
            self.models.append(copy.deepcopy(self.SVM))
            self.X_ind.append(self.cv.train[i])
    
    
    def _train_models(self):
        for i in range(self.num_models):
            if self.separate_feature_sets | self.separate_parameters:
                kernel_matrix = self._get_kernel_matrix(self.X_ind[i],self.X_ind[i],model_index = i)
            else:
                kernel_matrix = self._get_kernel_matrix(self.X_ind[i],self.X_ind[i])
                
            self.models[i].fit(kernel_matrix, self.cv.y[self.X_ind[i]])

    
    def _update_kernel_matrix(self):
        if self.separate_feature_sets | self.separate_parameters | self.sparse_kernel_matrix:
            for i in range(self.num_models):     
                if self.separate_feature_sets:
                    features = self.features[i]
                else:
                    features = self.features
                    
                if isinstance(self.parameters_,list):
                    parameters = self.parameters_[i].kernel
                else:
                    parameters = self.parameters_.kernel
                    
                if self.sparse_kernel_matrix:
                    training_kernel = self.kernel.compute(self.cv.X[self.cv.train[i], :], 
                                                          feature_index = features, 
                                                          parameters = parameters)
                
                    for j in range(len(self.cv.train[i])):
                        if self.separate_feature_sets | self.separate_parameters:
                            self.kernel_matrix_[i][self.cv.train[i][j],self.cv.train[i]] = training_kernel[j,:]
                        else:
                            self.kernel_matrix_[self.cv.train[i][j],self.cv.train[i]] = training_kernel[j,:]

                    testing_kernel = self.kernel.compute(self.cv.X[self.cv.test[i], :], 
                                                         feature_index = features, 
                                                         parameters = parameters,
                                                         Y = self.cv.X[self.cv.train[i], :])
                
                    for j in range(len(self.cv.test[i])):
                        if self.separate_feature_sets | self.separate_parameters:
                            self.kernel_matrix_[i][self.cv.test[i][j],self.cv.train[i]] = testing_kernel[j,:]
                        else:
                            self.kernel_matrix_[self.cv.test[i][j],self.cv.train[i]] = testing_kernel[j,:]
                        
                else:
                    self.kernel_matrix_[i] = self.kernel.compute(self.cv.X, 
                                                             feature_index = features, 
                                                             parameters = parameters)
        else:
            self.kernel_matrix_ = self.kernel.compute(self.cv.X, 
                                                      feature_index = self.features, 
                                                      parameters = self.parameters_.kernel)

    
    def _reset_kernel_matrix(self):
        if self.sparse_kernel_matrix:
            empty_matrix = dok_matrix((self.num_samples, self.num_samples))
        else: 
            empty_matrix = np.zeros((self.num_samples, self.num_samples))
            
        if self.separate_feature_sets | self.separate_parameters:
            self.kernel_matrix_ = []      
            for i in range(self.num_models):
                self.kernel_matrix_.append(copy.deepcopy(empty_matrix))
        else:
            self.kernel_matrix_ = empty_matrix

    
    def _get_kernel_matrix(self,indices_1, indices_2, model_index = None):
        if isinstance(self.kernel_matrix_,list):
            kernel_matrix = self.kernel_matrix_[model_index][indices_1, :][:, indices_2]
        else:
            kernel_matrix = self.kernel_matrix_[indices_1, :][:, indices_2]

        if self.sparse_kernel_matrix:
            returned_matrix = np.asarray(kernel_matrix.todense())
        else:
            returned_matrix = kernel_matrix

        return returned_matrix

    
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
                        best_kernel_matrix[i] = copy.deepcopy(self.kernel_matrix_)
                        best_parameters[i] = parameter_set
                        best_performance[i] = self.performance_[i]
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
        
        self._update_parameters(best_parameters)
        self.kernel_matrix_ = best_kernel_matrix

    
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
        

    def _update_parameters(self, parameter_set):
        self.parameters_ = parameter_set
        
        if isinstance(parameter_set,list):
            for i in range(self.num_models):
                for model_param in parameter_set[i].model.keys():
                        setattr(self.models[i],model_param, parameter_set[i].model[model_param])
        else:
            for i in range(self.num_models):
                for model_param in parameter_set.model.keys():
                        setattr(self.models[i],model_param, parameter_set.model[model_param])

        self._reset_kernel_matrix()
        self._update_kernel_matrix()

    
    def _remove_features(self, to_remove, model_index = None):
        self._reset_kernel_matrix()

        if model_index is not None:
            current_features = self.features[model_index]
        else:
            current_features = self.features
                
        for f in to_remove:
            ind_to_remove =  np.where(current_features == f)
            current_features = np.delete(current_features, ind_to_remove)
        
        if model_index is not None:
            self.features[model_index] = current_features
            self.removed_features_[model_index] = np.append(self.removed_features_[model_index], to_remove)
        else:
            self.features = current_features
            self.removed_features_ = np.append(self.removed_features_,to_remove)
            
        if len(current_features) > 0:
            self._update_kernel_matrix()
            
    
    def greedy_backward_selection(self, parameter_grid, 
                                  reduction_factor = 0.1, 
                                  feature_ranker = combined_rank().compute, 
                                  set_for_rank = "train"):
        
        feature_performance = {}
        result = 0
        best_score = -1e12
        if self.separate_feature_sets:
            n_feats = len(self.features[0])
        else:
            n_feats = len(self.features)
            
        while(n_feats >= 2) :
            self.tune_models(parameter_grid)

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
                    n_to_remove = np.floor(len(self.features[i])*reduction_factor).astype(int)
            
                    to_remove = self.features[i][feature_rank <= n_to_remove]
                    self._remove_features(to_remove, model_index = i)

                n_feats = len(self.features[0])
            else:
                rank_total = np.zeros(len(self.features))
                for i in range(self.num_models):
                    rank_total = rank_total + feature_ranker(self,i,set_for_rank)
                
                consensus_rank = rank_items(rank_total)
                n_to_remove = np.floor(len(self.features)*reduction_factor).astype(int)
        
                to_remove = self.features[consensus_rank <= n_to_remove]
                self._remove_features(to_remove)
                n_feats = len(self.features)
        
        if n_feats > 0:
            self.tune_models(parameter_grid)
            
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

    
    def feature_importance_(self, model_index):
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

        K = self.kernel.compute(support_vectors, 
                                feature_index = current_features, 
                                parameters = parameters)

        criteria = np.zeros(len(current_features))
        for z in range(len(current_features)):
            features_z = np.delete(current_features, z)
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
        
        decision_perturbation = np.zeros([len(X), len(current_features)])
        for z in range(len(current_features)):
            features_z = np.delete(current_features, z)
            Kp = self.kernel.compute(support_vectors, 
                                     feature_index = features_z, 
                                     parameters = parameters, 
                                     Y = X)
            
            decision_product = np.transpose(np.tile(self.models[model_index].dual_coef_[0,:], (len(X),1)))*(K-Kp)
            decision_perturbation[:,z] = np.array([sum(decision_product[:,i]) for i in range(decision_product.shape[1])])
                        
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
            
            decision_product = np.transpose(np.tile(self.models[model_index].dual_coef_[0,:], (len(X),1)))*dK
            decision_gradient[:,j] = np.array([sum(decision_product[:,i]) for i in range(decision_product.shape[1])])

        return decision_gradient

    
    def _find_boundary_points(self, model_index, X):
        boundary_points = np.zeros([len(X), self.cv.X.shape[1]])
        for i in range(0,len(X)):
            opt = minimize(svc_dec2, X[i,:], args=(self,model_index))
            boundary_points[i,:] = opt.x
            
        return boundary_points

    
    def integrated_gradient(self, X, model_index = None, num_steps = 20, ref_point = []):
        if isinstance(self.SVM, SVR) & (len(ref_point) == 0):
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
            if len(ref_point) == 0:
                ref_points = self._find_boundary_points(m,X)
        
            for i in range(0,len(X)):
                if len(ref_point) == 0:
                    x_start = ref_points[i,:]
                else:
                    x_start = ref_point
                
                xi = X[i,:]
                x_diff = xi - x_start
                
                x_steps = np.tile(x_start, (num_steps,1)) + np.tile(x_diff, (num_steps,1)) \
                            *np.transpose(np.tile(np.linspace(0, 1,num_steps), (self.cv.X.shape[1],1)))

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
                x = [self.feature_performance_[key]['num_features'] for key in self.feature_performance_.keys()]
                y = [self.feature_performance_[key][metric] for key in self.feature_performance_.keys()]
                
                area = np.trapz(y,x)
                enrichment_score = -area/max(x)
            
            case "max":
                y = [self.feature_performance_[key][metric] for key in self.feature_performance_.keys()]
                enrichment_score = max(y)
                
        return enrichment_score
        
