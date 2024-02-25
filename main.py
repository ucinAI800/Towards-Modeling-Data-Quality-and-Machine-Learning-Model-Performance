# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 00:10:13 2023

@author: 15135
"""

"""
Program that executes the experiment for paper 'Towards Modelling Trustworthiness in Machine Learning using DDR'.
To run, make sure to create an environment and install numpy, sci-kit learn, pytorch, torcheval, skorch, tqdm, scipy, and pandas, according to the directions
on the associated websites.
"""

import numpy as np
from sklearn import config_context as sklearn_config_context, datasets as sklearn_datasets, utils as sklearn_utils, preprocessing as sklearn_preprocessing, linear_model as sklearn_linear_model, svm as sklearn_svm, neighbors as sklearn_neighbors, ensemble as sklearn_ensemble, tree as sklearn_tree, cluster as sklearn_cluster, kernel_ridge as sklearn_kernel_ridge, model_selection as sklearn_model_selection, pipeline as sklearn_pipeline, base as sklearn_base, metrics as sklearn_metrics
import torch
from torcheval import metrics as torcheval_metrics
import skorch
import tqdm
import scipy
import math
#from matplotlib import pyplot as plt
#import seaborn as sns
import pandas as pd

with sklearn_config_context(assume_finite=True):
    # set random states/seeds for reproducibility
    RANDOM_STATE=0
    rng=np.random.default_rng(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    
    # DDR parameters
    # matrix of features DDR parameters
    N_FEATURE_MATRIX_DDRS=100 #int(input("Enter the number of matrix of features DDRs: "))   # number of matrix of features DDRs
    LOW_FEATURE_MATRIX_DDR=1/N_FEATURE_MATRIX_DDRS   #minimum matrix of features DDR
    FEATURE_MATRIX_DDRS=np.linspace(LOW_FEATURE_MATRIX_DDR, 1, N_FEATURE_MATRIX_DDRS)
    # set of feature DDRs parameters
    MAX_N_SETS_FEATURE_DDRS=100   # maximum number of sets of feature DDRs (number of sets of feature DDRs for matrix of features DDR == 1/2)
    NS_SETS_FEATURE_DDRS=np.zeros(N_FEATURE_MATRIX_DDRS, dtype=int)    # numbers of sets of feature DDRs for each matrix of features DDR
    for i, FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
        # number of sets of feature DDRs for given matrix of features DDR
        #formula for upper half of ellipse with points (0, 0), (1/2, MAX_N_SETS_FEATURE_DDRS), and (1, 0)
        N_SETS_FEATURE_DDRS=int(math.ceil(1+2*(MAX_N_SETS_FEATURE_DDRS-1)*(FEATURE_MATRIX_DDR-FEATURE_MATRIX_DDR**2)**(1/2)))
        NS_SETS_FEATURE_DDRS[i]=N_SETS_FEATURE_DDRS    
    # number of sets of feature DDRs to skip when calculating all sets of feature DDRs.
    #The higher this number, the more independent the sampled sets of feature DDRs become.
    THIN=2
            
    
    #All model/dataset parameters are default, unless otherwise specified
    # KERNEL_RIDGE_REGRESSION_ALPHA=1
    # KERNEL_RIDGE_REGRESSION_KERNEL="sigmoid" #nondefault
    # KERNEL_RIDGE_REGRESSION_GAMMA=None
    # KERNEL_RIDGE_REGRESSION_COEF0=1
    
    # KERNEL_RIDGE_REGRESSION_FRIEDMAN1_N_SAMPLES=100
    # KERNEL_RIDGE_REGRESSION_FRIEDMAN1_N_FEATURES=10
    # KERNEL_RIDGE_REGRESSION_FRIEDMAN1_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    ORDINARY_LEAST_SQUARES_RANDOM_REGRESSION_N_SAMPLES=1000 #int(input("Enter the number of samples for random regression for ordinary least squares: ")) #default is 100
    ORDINARY_LEAST_SQUARES_RANDOM_REGRESSION_N_FEATURES=1 #int(input("Enter the number of features for random regression for ordinary least squares: ")) # default is 100
    ORDINARY_LEAST_SQUARES_RANDOM_REGRESSION_N_INFORMATIVE=ORDINARY_LEAST_SQUARES_RANDOM_REGRESSION_N_FEATURES    #default is 10
    ORDINARY_LEAST_SQUARES_RANDOM_REGRESSION_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    # LASSO_ALPHA=1
    
    # LASSO_RANDOM_REGRESSION_SPARSE_UNCORRELATED_N_SAMPLES=100
    # LASSO_RANDOM_REGRESSION_SPARSE_UNCORRELATED_N_FEATURES=10
    # LASSO_RANDOM_REGRESSION_SPARSE_UNCORRELATED_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    # ELASTIC_NET_ALPHA=1
    # ELASTIC_NET_L1_RATIO=0.5
    
    # ELASTIC_NET_SINUSOIDAL_SPARSE_CORRELATED_N_SAMPLES=50
    # ELASTIC_NET_SINUSOIDAL_SPARSE_CORRELATED_N_FEATURES=100
    # ELASTIC_NET_SINUSOIDAL_SPARSE_CORRELATED_N_INFORMATIVE=10
    
    
    DECISION_TREE_CLASSIFIER_MAX_DEPTH=3    #nondefault
    DECISION_TREE_CLASSIFIER_MIN_SAMPLES_LEAF=1    #nondefault
    DECISION_TREE_CLASSIFIER_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_SAMPLES=1000 #int(input("Enter the number of samples for random 2-class classification for decision tree classifier: ")) #deault=100
    DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES=1 #int(input("Enter the number of features for random 2-class classification for decision tree classifier: ")) #default=20
    DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_INFORMATIVE=DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES #default=2
    DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_REDUNDANT=0 #int(input("Enter the number of redundant features for random 2-class classification for decision tree classifier: ")) #deault=2
    DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLASSES=2
    DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLUSTERS_PER_CLASS=1 #default=2
    DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    DECISION_TREE_REGRESSOR_MAX_DEPTH=3    #nondefault
    DECISION_TREE_REGRESSOR_MIN_SAMPLES_LEAF=5    #nondefault
    DECISION_TREE_REGRESSOR_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    DECISION_TREE_REGRESSOR_FRIEDMAN1_N_SAMPLES=1000 #int(input("Enter the number of samples for friedman #1 for decision tree regressor: "))  #default=100
    DECISION_TREE_REGRESSOR_FRIEDMAN1_N_FEATURES=5 #int(input("Enter the number of features for friedman #1 decision tree regressor: "))  #default=10
    DECISION_TREE_REGRESSOR_FRIEDMAN1_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    # RIDGE_REGRESSOR_ALPHA=1
    # RIDGE_REGRESSOR_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    # RIDGE_REGRESSOR_RANDOM_REGRESSION_N_SAMPLES=100
    # RIDGE_REGRESSOR_RANDOM_REGRESSION_N_FEATURES=100
    # RIDGE_REGRESSOR_RANDOM_REGRESSION_N_INFORMATIVE=10
    # RIDGE_REGRESSOR_RANDOM_REGRESSION_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    # RIDGE_CLASSIFIER_ALPHA=1
    # RIDGE_CLASSIFIER_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    # RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_SAMPLES=100
    # RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES=20
    # RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_INFORMATIVE=2
    # RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_REDUNDANT=2
    # RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLASSES=2
    # RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLUSTERS_PER_CLASS=2
    # RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    BINARY_LOGISTIC_REGRESSION_C=1
    
    BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_N_SAMPLES=1000 #int(input("Enter the number of samples for random 2-class classification for binary logistic regression: ")) #defalut=100
    BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES=1
    BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_N_INFORMATIVE=1
    BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_N_REDUNDANT=0
    BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_N_CLASSES=2
    BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_N_CLUSTERS_PER_CLASS=1
    BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    LINEAR_SUPPORT_VECTOR_CLASSIFIER_C=1
    LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_SAMPLES=1000 #int(input("Enter the number of samples for random 2-class classification for linear support vector classifier: ")) #defalut=100
    LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES=1
    LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_INFORMATIVE=1
    LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_REDUNDANT=0
    LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLASSES=2
    LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLUSTERS_PER_CLASS=1
    LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    LINEAR_SUPPORT_VECTOR_REGRESSOR_EPSILON=0
    LINEAR_SUPPORT_VECTOR_REGRESSOR_C=1
    LINEAR_SUPPORT_VECTOR_REGRESSOR_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    LINEAR_SUPPORT_VECTOR_REGRESSOR_RANDOM_REGRESSION_N_SAMPLES=1000#int(input("Enter the number of samples for random regression for linear support vector regressor: ")) #default=100
    LINEAR_SUPPORT_VECTOR_REGRESSOR_RANDOM_REGRESSION_N_FEATURES=1 #int(input("Enter the number of features for random regression for linear support vector regressor: ")) #default=100
    LINEAR_SUPPORT_VECTOR_REGRESSOR_RANDOM_REGRESSION_N_INFORMATIVE=LINEAR_SUPPORT_VECTOR_REGRESSOR_RANDOM_REGRESSION_N_FEATURES #default=10
    LINEAR_SUPPORT_VECTOR_REGRESSOR_RANDOM_REGRESSION_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    K_NEAREST_NEIGHBORS_CLASSIFICATION_N_NEIGHBORS=5
    
    K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_N_SAMPLES=1000#int(input("Enter the number of samples for random 2-class classification for k-nearest neighbors classifier: ")) #defalut=100
    K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES=1
    K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_N_INFORMATIVE=1
    K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_N_REDUNDANT=0
    K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_N_CLASSES=2
    K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_N_CLUSTERS_PER_CLASS=1
    K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    K_NEAREST_NEIGHBORS_REGRESSION_N_NEIGHBORS=5
    
    K_NEAREST_NEIGHBORS_REGRESSION_FRIEDMAN1_N_SAMPLES=1000#int(input("Enter the number of samples for friedman #1 for k nearest neighbors regression: ")) #100
    K_NEAREST_NEIGHBORS_REGRESSION_FRIEDMAN1_N_FEATURES=5 #int(input("Enter the number of features for friedman #1 for k nearest neighbors regression: "))#10
    K_NEAREST_NEIGHBORS_REGRESSION_FRIEDMAN1_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    # RANDOM_FOREST_CLASSIFIER_N_ESTIMATORS=100
    # RANDOM_FOREST_CLASSIFIER_MAX_FEATURES='sqrt'
    # RANDOM_FOREST_CLASSIFIER_MAX_DEPTH=None
    # RANDOM_FOREST_CLASSIFIER_MIN_SAMPLES_SPLIT=2
    # RANDOM_FOREST_CLASSIFIER_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    # RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_SAMPLES=100
    # RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES=20
    # RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_INFORMATIVE=2
    # RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_REDUNDANT=2
    # RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLASSES=2
    # RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLUSTERS_PER_CLASS=2
    # RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    # RANDOM_FOREST_REGRESSOR_N_ESTIMATORS=100
    # RANDOM_FOREST_REGRESSOR_MAX_FEATURES=None    #nondefault
    # RANDOM_FOREST_REGRESSOR_MAX_DEPTH=None
    # RANDOM_FOREST_REGRESSOR_MIN_SAMPLES_SPLIT=2
    # RANDOM_FOREST_REGRESSOR_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    # RANDOM_FOREST_REGRESSOR_FRIEDMAN1_N_SAMPLES=int(input("Enter the number of samples for friedman #1 for random forest regressor: "))#100
    # RANDOM_FOREST_REGRESSOR_FRIEDMAN1_N_FEATURES=int(input("Enter the number of features for friedman #1 for random forest regressor: "))#10
    # RANDOM_FOREST_REGRESSOR_FRIEDMAN1_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    MULTI_LAYER_PERCEPTRON_CLASSIFIER_LAYER_WIDTHS=(1, 100, 2)
    
    MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_SAMPLES=1000#int(input("Enter the number of samples for random 2-class classification for multi-layer perceptron classifier: ")) #defalut=100
    MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES=1
    MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_INFORMATIVE=1
    MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_REDUNDANT=0
    MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLASSES=2
    MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLUSTERS_PER_CLASS=1
    MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    MULTI_LAYER_PERCEPTRON_REGRESSOR_LAYER_WIDTHS=(5, 100, 1)
    
    MULTI_LAYER_PERCEPTRON_REGRESSOR_FRIEDMAN1_N_SAMPLES=1000#int(input("Enter the number of samples for friedman #1 for multi-layer perceptron regressor: "))#100
    MULTI_LAYER_PERCEPTRON_REGRESSOR_FRIEDMAN1_N_FEATURES=5 #int(input("Enter the number of features for friedman #1 for multi-layer perceptron regressor: "))#10
    MULTI_LAYER_PERCEPTRON_REGRESSOR_FRIEDMAN1_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    # QUADRATIC_REGRESSION_RANDOM_QUADRATIC_REGRESSION_N_SAMPLES=int(input("Enter the number of samples for random quadratic regression for quadratic regression: ")) #default=100
    # QUADRATIC_REGRESSION_RANDOM_QUADRATIC_REGRESSION_N_FEATURES=int(input("Enter the number of features for random quadratic regression for quadratic regression: ")) #default=100
    # QUADRATIC_REGRESSION_RANDOM_QUADRATIC_REGRESSION_N_INFORMATIVE=int(input("Enter the number of informative features for random quadratic regression for quadratic regression: ")) #default=10
    # QUADRATIC_REGRESSION_RANDOM_QUADRATIC_REGRESSION_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    K_MEANS_N_CLUSTERS=2    #nondefault
    K_MEANS_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    K_MEANS_ISOTROPIC_GAUSSIAN_BLOBS_N_SAMPLES=1000#int(input("Enter the number of samples for isotropic Gaussian blobs for k means: "))  #default=100
    K_MEANS_ISOTROPIC_GAUSSIAN_BLOBS_N_FEATURES=1 #int(input("Enter the number of features for isotropic Gaussian blobs for k means: ")) #default=2
    K_MEANS_ISOTROPIC_GAUSSIAN_BLOBS_CENTERS=2 #nondefault
    K_MEANS_ISOTROPIC_GAUSSIAN_BLOBS_CLUSTER_STD=1/N_FEATURE_MATRIX_DDRS #default=1
    K_MEANS_ISOTROPIC_GAUSSIAN_BLOBS_RANDOM_STATE=RANDOM_STATE    #nondefault
    
    
    def Parameters():
        """
        Returns the parameters for the experiment
    
        Returns
        -------
        parameters: Dict
            Parameters for the experiment.
    
        """
          
        def Learning_Parameters(learning_type):
            """
            Takes a learning type and returns the associated parameters
    
            Parameters
            ----------
            learning_type : string
                Type of learning.
    
            Returns
            -------
            learning_parameters: Dict
                Parameters for type of learning learning_type.
    
            """
            
            def Model_Parameters(model_type):
                """
                Takes a model type and returns the parameters for it.
    
                Parameters
                ----------
                model_type : string
                    Type of model.
    
                Returns
                -------
                model_parameters : Dict
                    Parameters of the model type.
    
                """
    
                def Submodel_Parameters(submodel_type):
                    """
                    Take a type of submodel and return its parameters.
    
                    Parameters
                    ----------
                    submodel_type : string
                        The type of submodel.
    
                    Returns
                    -------
                    submodel_parameters : Dict
                        Parameters for the submodel.
    
                    """
    
                    def Sets_Feature_DDRs(feature_matrix_ddr, n_sets_feature_ddrs, n_features):
                        """
                        Takes the matrix of features DDR, the number of sets of feature DDRs, and number of features and returns the sets of feature DDRs.
    
                        Parameters
                        ----------
                        feature_matrix_ddr : Float
                            The matrix of features DDR.
                        n_sets_feature_ddrs : Int
                            The number of sets of feature DDRs.
                        n_features : Int
                            The number of features in the matrix of features.
    
                        Returns
                        -------
                        sets_feature_DDRs : np.ndarray of Floats of size (n_sets_feature_ddrs, n_features)
                            The sets of feature DDRs.
    
                        """
                        
                        """
                        A class to hold polytopes in H-representation.
    
                        Francesc Font-Clos
                        Oct 2018
                        """
    
    
                        class Polytope(object):
                            """A polytope in H-representation."""
                            
                            def __init__(self):
                                """
                                Create a polytope in H-representation.
    
                                The polytope is defined as the set of
                                points x in Rn such that
    
                                A x <= b
    
                                """
                            
                                def A(feature_matrix_ddr, n_features):
                                    """
                                    Takes the matrix of features DDR and number of features and returns the constraint matrix.
    
                                    Parameters
                                    ----------
                                    feature_matrix_ddr : Float
                                        The matrix of features DDR.
                                    n_features : Int
                                        The number of features.
    
                                    Returns
                                    -------
                                    a : np.ndarray of Ints of shape (2*n_features, n_features - 1)
                                        The matrix of constraints on the feature DDRs.
    
                                    """
                                    
                                    # number of constraints=number of inequalities=2*number of features
                                    #degrees of freedom = number of features -1
                                    #since feature DDR 1 + ... + feature DDR (number of features -1) == number of features*matrix of features DDR - feature DDR (number of features)
                                    a=np.zeros((2*n_features, n_features-1))    
                                    
                                    for i in range(n_features-1):
                                        
                                        #row representing the constraint -1*feature DDR i <= 0
                                        a_row_type_1=np.zeros(n_features-1)
                                        #row representing the constraint 1*feature DDR i <= 1
                                        a_row_type_2=np.zeros(n_features-1)
                                        
                                        a_row_type_1[i]=-1
    
                                        a_row_type_2[i]=1
                                        
                                        a[i]=a_row_type_1
    
                                        a[n_features-1+i]=a_row_type_2
    
                                    #row representing the constraint
                                    #-1*feature DDR 1 - ... -1*feature DDR (number of features -1) <= 1 - number of features * matrix of features DDR  
                                    a[2*n_features-2]=-np.ones(n_features-1)
                                    #row representing the constraint 
                                    #1*feature DDR 1 + ... + 1*feature DDR (number of features -1) <= number of features*matrix of features DDR 
                                    a[2*n_features-1]=np.ones(n_features-1)        
                                            
                                    return a
    
                                def B(feature_matrix_ddr, n_features):
                                    """
                                    Takes the matrix of features DDR and the number of features and returns the constraint vector.
    
                                    Parameters
                                    ----------
                                    feature_matrix_ddr : Float
                                        Matrix of features DDR.
                                    n_features : Int
                                        Number of features.
    
                                    Returns
                                    -------
                                    b : np.array of Floats of size 2*n_features
                                        Constraint vector.
    
                                    """
                                
                                    # number of constraints=number of inequalities=2*number of features
                                    b=np.zeros(2*n_features)
                                    
                                    for i in range(n_features-1):
                                        
                                        #value representing the constraint -1*feature DDR i <= 0
                                        b[i]=0
                                        #value representing the constraint feature DDR i <= 1
                                        b[n_features-1+i]=1
                                        
                                     #row representing the constraint
                                     #-1*feature DDR 1 - ... -1*feature DDR (number of features -1) <= 1 - number of features * matrix of features DDR  
                                    b[2*n_features-2]=1-n_features*feature_matrix_ddr
                                    
                                    #row representing the constraint 
                                    #1*feature DDR 1 + ... + 1*feature DDR (number of features -1) <= number of features*matrix of features DDR 
                                    b[2*n_features-1]=n_features*feature_matrix_ddr
                                    
                                    return b
                                
                                a=A(feature_matrix_ddr, n_features)
                                b=B(feature_matrix_ddr, n_features)
                                # dimensionality verifications
                                assert a is not None and b is not None
                                assert len(b.shape) == 1
                                assert len(a.shape) == 2
                                assert a.shape[0] == len(b)
                                # store data
                               
                                self.A = a
                                self.b = b
                                self.dim = a.shape[1]
                                self.nplanes = a.shape[0]
                                self._find_auxiliar_points_in_planes()
    
                            def check_inside(self, point):
                                """Check if a point is inside the polytope."""
                                checks = self.A@point <= self.b
                                check = np.all(checks)
                                return check
    
                            def _find_auxiliar_points_in_planes(self):
                                """Find an auxiliar point for each plane."""
                                aux_points = [self._find_auxiliar_point(self.A[i],
                                                                        self.b[i])
                                              for i in range(self.nplanes)]
                                self.auxiliar_points = aux_points
    
                            def _find_auxiliar_point(self, Ai, bi):
                                """Find an auxiliar point for one plane."""
                                p = np.zeros(self.dim)
                                j = np.argmax(Ai != 0)
                                p[j] = bi / Ai[j]
                                return p
    
    
                        """
                        MinOver algorithm to find a point inside a polytope.
    
                        Francesc Font-Clos
                        Oct 2018
                        """
    
    
                        class MinOver(object):
                            """MinOver solver."""
    
                            def __init__(self, polytope, ):
                                """
                                Create a MinOver solver.
    
                                Parameters
                                ----------
                                polytope: hitandrun.polytope
                                    Polytope in H-representation
    
                                """
                                self.polytope = polytope
    
                            def run(self, speed=1, starting_point=None, max_iters=100, verbose=False):
                                """
                                Run the MinOver algorithm.
    
                                Parameters
                                ----------
                                speed: float
                                    Distance moved at each learning step
                                max_iters: int
                                    Maximum number of iterations (per hyperplan).
                                starting_poin: np.array
                                    Initial condition.
    
                                Returns
                                -------
                                current: np.array
                                    The final point.
                                convergence: bool
                                    True if the algorithm converged, False, otherwise.
    
                                """
                                self.max_iters = max_iters * self.polytope.nplanes
                                self.speed = speed
                                if starting_point is None:
                                    self.current = np.zeros(self.polytope.dim)
                                else:
                                    self.current = starting_point
                                # compute step 0 worst planes
                                # this is a trick to handle first steps
                                self.worst_indexes = [-1, -2]
                                self.worst_distances = [-1, -2]
                                self._set_worst_constraint()
                                for i in range(self.max_iters):
                                    convergence = self._step()
                                    self.iter = i
                                    self._check_speed()
                                    if verbose:
                                        self._print_worst()
                                    if convergence:
                                        break
                                return self.current, convergence
    
                            def _step(self):
                                self._move_towards_worst_plane()
                                self._set_worst_constraint()
                                return np.all(self.distances < 0)
    
                            def _check_speed(self):
                                i0, i1, i2 = self.worst_indexes[::-1][:3]
                                d0, d1, d2 = self.worst_distances[::-1][:3]
                                if i0 != i1 and i0 == i2 and d0 >= d2:
                                    self.speed *= 0.9
    
                            def _set_worst_constraint(self):
                                self.distances = self.polytope.A @ self.current - self.polytope.b
                                self.worst = np.argmax(self.distances)
                                self.worst_indexes.append(self.worst)
                                self.worst_distances.append(self.distances[self.worst])
    
                            def _move_towards_worst_plane(self):
                                self.current = self.current - self.speed * self.polytope.A[self.worst]
    
                            def _print_worst(self):
                                worst_distance = self.distances[self.worst]
                                print("iter", self.iter,
                                      "index:", self.worst,
                                      "distance:", worst_distance,
                                      "speed:", self.speed)
    
    
                        """
                        Hit-and-run sampler.
    
                        Francesc Font-Clos
                        Oct 2018
                        """
    
    
                        class HitAndRun(object):
                            """Hit-and-run sampler."""
    
                            def __init__(self, polytope=None, starting_point=None,
                                         n_samples=100, thin=1):
                                """
                                Create a hit-and-run sampler.
    
                                Parameters
                                ----------
                                polytope: hitandrun.polytope
                                    The convex polytope to be sampled.
                                starting_point: np.array
                                    Initial condition. Must be inside the polytope.
                                n_samples: int
                                    Number of desired samples.
                                thin : int
                                    Thinning factor, increase to get independent samples.
    
                                """
                                # make sure we got a point inside the polytope
                                assert starting_point is not None
                                assert len(starting_point) == polytope.dim
                                assert polytope.check_inside(starting_point)
    
                                self.n_features=polytope.dim+1
                                self.polytope = polytope
                                self.starting_point = starting_point
                                self.n_samples = n_samples
                                self.thin = thin
                                # place starting point as current point
                                self.current = starting_point
                                # set a starting random direction
                                self._set_random_direction()
                                # create empty list of samples
                                self.samples = np.zeros((n_samples, self.n_features))
    
                            def get_samples(self):
                                """Get the requested samples."""
                                #self.samples = []    #ignore since self.samples is an array
    
                                # keep only one every thin
                                for i in range(self.n_samples):
                                    for _ in range(self.thin):
                                        self._step()
                                    self._add_current_to_samples(i)
                                return self.samples
    
                            # private functions
                            def _step(self):
                                """Make one step."""
                                # set random direction
                                self._set_random_direction()
                                # find lambdas
                                self._find_lambdas()
                                # find smallest positive and negative lambdas
                                try:
                                    lam_plus = np.min(self.lambdas[self.lambdas > 0])
                                    lam_minus = np.max(self.lambdas[self.lambdas < 0])
                                except(Exception):
                                    raise RuntimeError("The current direction does not intersect"
                                                       "any of the hyperplanes.")
                                # throw random point between lambdas
                                lam = rng.uniform(low=lam_minus, high=lam_plus)
                                # compute new point and add it
                                new_point = self.current + lam * self.direction
                                self.current = new_point
    
                            def _find_lambdas(self):
                                """
                                Find the lambda value for each hyperplane.
    
                                The lambda value is the distance we have to travel
                                in the current direction, from the current point, to
                                reach a given hyperplane.
                                """
                                A = self.polytope.A
                                p = self.polytope.auxiliar_points
    
                                lambdas = []
                                for i in range(self.polytope.nplanes):
                                    if np.isclose(self.direction @ A[i], 0):
                                        lambdas.append(np.nan)
                                    else:
                                        lam = ((p[i] - self.current) @ A[i]) / (self.direction @ A[i])
                                        lambdas.append(lam)
                                self.lambdas = np.array(lambdas)
    
                            def _set_random_direction(self):
                                """Set a unitary random direction in which to travel."""
                                direction = rng.standard_normal(self.polytope.dim)
                                self.direction = direction / scipy.spatial.distance.norm(direction)
    
                            def _add_current_to_samples(self, i):
                                for j in range(self.n_features-1):
                                    feature=self.current[j]
                                    self.samples[i][j]=feature
                                features=self.samples[i]
                                feature=feature_matrix_ddr*self.n_features-np.sum(features)
                                self.samples[i][self.n_features-1]=feature
                                rng.shuffle(self.samples[i])
    
    
                        if not feature_matrix_ddr==1 and not n_features==1:    
                            polytope=Polytope()
                            starting_point=np.zeros(n_features-1)
                            for i in range(n_features-1):
                                starting_point[i]=feature_matrix_ddr
                            hit_and_run=HitAndRun(polytope, starting_point, n_samples=n_sets_feature_ddrs, thin=THIN)
                            sets_feature_DDRs=hit_and_run.get_samples()
                            
                        else:
                            sets_feature_DDRs=feature_matrix_ddr*np.ones((1,n_features))
                        return sets_feature_DDRs
                    
                    def Sub2model_Parameters(sub2model_type):
                        """
                        Takes a type of sub2model and returns its parameters.
    
                        Parameters
                        ----------
                        sub2model_type : string
                            Type of sub2model.
    
                        Returns
                        -------
                        sub2model_parameters : Dict
                            Parameters for the sub2model.
    
                        """
                        
                        def Sub3model_Parameters(sub3model_type):
                            """
                            Takes a type of sub3model and returns its parameters.
    
                            Parameters
                            ----------
                            sub3model_type : string
                                The type of sub3model.
    
                            Returns
                            -------
                            sub3model_parameters : Dict
                                The parameters for the sub3model.
    
                            """
                            
                            sub3model_parameters={}
                            match sub3model_type:
                                # case "ridge regressor random regression":
                                #     sub3model_parameters["number of features"]=RIDGE_REGRESSOR_RANDOM_REGRESSION_N_FEATURES
                                #     sub3model_parameters["number of informative features"]=RIDGE_REGRESSOR_RANDOM_REGRESSION_N_INFORMATIVE
                                #     sub3model_parameters["number of samples"]=RIDGE_REGRESSOR_RANDOM_REGRESSION_N_SAMPLES
                                #     sub3model_parameters["random state"]=RIDGE_REGRESSOR_RANDOM_REGRESSION_RANDOM_STATE
                                #     for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                                #         N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                #         SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, RIDGE_REGRESSOR_RANDOM_REGRESSION_N_FEATURES)
                                #         sub3model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS                                
                                # case "ridge classifier random 2-class classification":
                                #     sub3model_parameters["number of classes"]=RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLASSES
                                #     sub3model_parameters["number of clusters per class"]=RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLUSTERS_PER_CLASS
                                #     sub3model_parameters["number of features"]=RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES
                                #     sub3model_parameters["number of informative features"]=RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_INFORMATIVE
                                #     sub3model_parameters["number of redundant features"]=RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_REDUNDANT
                                #     sub3model_parameters["number of samples"]=RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_SAMPLES
                                #     sub3model_parameters["random state"]=RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_RANDOM_STATE
                                #     for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                                #         N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                #         SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, RIDGE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES)
                                #         sub3model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS  
                                case "binary logistic regression random 2-class classification":
                                    sub3model_parameters["number of classes"]=BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_N_CLASSES
                                    sub3model_parameters["number of clusters per class"]=BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_N_CLUSTERS_PER_CLASS
                                    sub3model_parameters["number of features"]=BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES
                                    sub3model_parameters["number of informative features"]=BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_N_INFORMATIVE
                                    sub3model_parameters["number of redundant features"]=BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_N_REDUNDANT
                                    sub3model_parameters["number of samples"]=BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_N_SAMPLES
                                    sub3model_parameters["random state"]=BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_RANDOM_STATE
                                    for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                                        N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                        SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, BINARY_LOGISTIC_REGRESSION_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES)
                                        sub3model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS  
                                # case "quadratic regression random quadratic regression":
                                #     sub3model_parameters["number of features"]=QUADRATIC_REGRESSION_RANDOM_QUADRATIC_REGRESSION_N_FEATURES
                                #     sub3model_parameters["number of informative features"]=QUADRATIC_REGRESSION_RANDOM_QUADRATIC_REGRESSION_N_INFORMATIVE
                                #     sub3model_parameters["number of samples"]=QUADRATIC_REGRESSION_RANDOM_QUADRATIC_REGRESSION_N_SAMPLES
                                #     sub3model_parameters["random state"]=QUADRATIC_REGRESSION_RANDOM_QUADRATIC_REGRESSION_RANDOM_STATE
                                #     for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                                #         N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                #         SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, QUADRATIC_REGRESSION_RANDOM_QUADRATIC_REGRESSION_N_FEATURES)
                                #         sub3model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS
                                case "linear support vector classifier random 2-class classification":
                                    sub3model_parameters["number of classes"]=LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLASSES
                                    sub3model_parameters["number of clusters per class"]=LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLUSTERS_PER_CLASS
                                    sub3model_parameters["number of features"]=LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES
                                    sub3model_parameters["number of informative features"]=LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_INFORMATIVE
                                    sub3model_parameters["number of redundant features"]=LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_REDUNDANT
                                    sub3model_parameters["number of samples"]=LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_SAMPLES
                                    sub3model_parameters["random state"]=LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_RANDOM_STATE
                                    for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                                        N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                        SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES)
                                        sub3model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS
                                case "linear support vector regressor random regression":
                                    sub3model_parameters["number of features"]=LINEAR_SUPPORT_VECTOR_REGRESSOR_RANDOM_REGRESSION_N_FEATURES
                                    sub3model_parameters["number of informative features"]=LINEAR_SUPPORT_VECTOR_REGRESSOR_RANDOM_REGRESSION_N_INFORMATIVE
                                    sub3model_parameters["number of samples"]=LINEAR_SUPPORT_VECTOR_REGRESSOR_RANDOM_REGRESSION_N_SAMPLES
                                    sub3model_parameters["random state"]=LINEAR_SUPPORT_VECTOR_REGRESSOR_RANDOM_REGRESSION_RANDOM_STATE
                                    for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                                        N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                        SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, LINEAR_SUPPORT_VECTOR_REGRESSOR_RANDOM_REGRESSION_N_FEATURES)
                                        sub3model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS       
                                case "k nearest neighbors classifier random 2-class classification":
                                    sub3model_parameters["number of classes"]=K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_N_CLASSES
                                    sub3model_parameters["number of clusters per class"]=K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_N_CLUSTERS_PER_CLASS
                                    sub3model_parameters["number of features"]=K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES
                                    sub3model_parameters["number of informative features"]=K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_N_INFORMATIVE
                                    sub3model_parameters["random state"]=K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_RANDOM_STATE
                                    sub3model_parameters["number of redundant features"]=K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_N_REDUNDANT
                                    sub3model_parameters["number of samples"]=K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_N_SAMPLES
                                    for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                                        N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                        SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, K_NEAREST_NEIGHBORS_CLASSIFICATION_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES)
                                        sub3model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS       
                                case "k nearest neighbors regressor friedman #1":
                                    sub3model_parameters["number of features"]=K_NEAREST_NEIGHBORS_REGRESSION_FRIEDMAN1_N_FEATURES
                                    sub3model_parameters["number of samples"]=K_NEAREST_NEIGHBORS_REGRESSION_FRIEDMAN1_N_SAMPLES
                                    sub3model_parameters["random state"]=K_NEAREST_NEIGHBORS_REGRESSION_FRIEDMAN1_RANDOM_STATE
                                    for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                                        N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                        SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, K_NEAREST_NEIGHBORS_REGRESSION_FRIEDMAN1_N_FEATURES)
                                        sub3model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS       
                                # case "random forest classifier random 2-class classification":
                                #     sub3model_parameters["number of classes"]=RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLASSES
                                #     sub3model_parameters["number of clusters per class"]=RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLUSTERS_PER_CLASS
                                #     sub3model_parameters["number of features"]=RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES
                                #     sub3model_parameters["number of informative features"]=RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_INFORMATIVE
                                #     sub3model_parameters["random state"]=RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_RANDOM_STATE
                                #     sub3model_parameters["number of redundant features"]=RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_REDUNDANT
                                #     sub3model_parameters["number of samples"]=RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_SAMPLES
                                #     for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                                #         N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                #         SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, RANDOM_FOREST_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES)
                                #         sub3model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS       
                                # case "random forest regressor friedman #1":
                                #     sub3model_parameters["number of features"]=RANDOM_FOREST_REGRESSOR_FRIEDMAN1_N_FEATURES
                                #     sub3model_parameters["number of samples"]=RANDOM_FOREST_REGRESSOR_FRIEDMAN1_N_SAMPLES
                                #     sub3model_parameters["random state"]=RANDOM_FOREST_REGRESSOR_FRIEDMAN1_RANDOM_STATE
                                #     for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                                #         N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                #         SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, RANDOM_FOREST_REGRESSOR_FRIEDMAN1_N_FEATURES)
                                #         sub3model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS     
                                case "multi-layer perceptron classifier random 2-class classification":
                                    sub3model_parameters["number of classes"]=MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLASSES
                                    sub3model_parameters["number of clusters per class"]=MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLUSTERS_PER_CLASS
                                    sub3model_parameters["number of features"]=MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES
                                    sub3model_parameters["number of informative features"]=MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_INFORMATIVE
                                    sub3model_parameters["random state"]=MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_RANDOM_STATE
                                    sub3model_parameters["number of redundant features"]=MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_REDUNDANT
                                    sub3model_parameters["number of samples"]=MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_SAMPLES
                                    for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                                        N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                        SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, MULTI_LAYER_PERCEPTRON_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES)
                                        sub3model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS       
                                case "multi-layer perceptron regressor friedman #1":
                                    sub3model_parameters["number of features"]=MULTI_LAYER_PERCEPTRON_REGRESSOR_FRIEDMAN1_N_FEATURES
                                    sub3model_parameters["number of samples"]=MULTI_LAYER_PERCEPTRON_REGRESSOR_FRIEDMAN1_N_SAMPLES
                                    sub3model_parameters["random state"]=MULTI_LAYER_PERCEPTRON_REGRESSOR_FRIEDMAN1_RANDOM_STATE
                                    for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                                        N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                        SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, MULTI_LAYER_PERCEPTRON_REGRESSOR_FRIEDMAN1_N_FEATURES)
                                        sub3model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS       
                            return sub3model_parameters
                        
                        sub2model_parameters={}
                        sub3model_types=[]
                        match sub2model_type:
                            case "ordinary least squares random regression":
                                sub2model_parameters["number of samples"]=ORDINARY_LEAST_SQUARES_RANDOM_REGRESSION_N_SAMPLES
                                sub2model_parameters["number of features"]=ORDINARY_LEAST_SQUARES_RANDOM_REGRESSION_N_FEATURES
                                sub2model_parameters["number of informative features"]=ORDINARY_LEAST_SQUARES_RANDOM_REGRESSION_N_INFORMATIVE
                                sub2model_parameters["random state"]=ORDINARY_LEAST_SQUARES_RANDOM_REGRESSION_RANDOM_STATE
                                for i,FEATURE_MATRIX_DDR in tqdm.tqdm(enumerate(FEATURE_MATRIX_DDRS), desc=f"Generating {sub2model_type} sets of feature DDRs for each matrix of features DDR: "):
                                    N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                    SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, ORDINARY_LEAST_SQUARES_RANDOM_REGRESSION_N_FEATURES)
                                    sub2model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS       
                            # case "ridge regressor":
                            #     sub3model_types=["ridge regressor random regression"]
                            #     sub2model_parameters["alpha"]=RIDGE_REGRESSOR_ALPHA
                            #     sub2model_parameters["random state"]=RIDGE_REGRESSOR_RANDOM_STATE
                            # case "ridge classifier":
                            #     sub3model_types=["ridge classifier random 2-class classification"]
                            #     sub2model_parameters["alpha"]=RIDGE_CLASSIFIER_ALPHA
                            #     sub2model_parameters["random state"]=RIDGE_CLASSIFIER_RANDOM_STATE
                            # case "lasso random regression sparse uncorrelated":
                            #     sub2model_parameters["number of features"]=LASSO_RANDOM_REGRESSION_SPARSE_UNCORRELATED_N_FEATURES
                            #     sub2model_parameters["number of samples"]=LASSO_RANDOM_REGRESSION_SPARSE_UNCORRELATED_N_SAMPLES
                            #     sub2model_parameters["random state"]=LASSO_RANDOM_REGRESSION_SPARSE_UNCORRELATED_RANDOM_STATE
                            #     for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                            #         N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                            #         SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, LASSO_RANDOM_REGRESSION_SPARSE_UNCORRELATED_N_FEATURES)
                            #         sub2model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS       
                            # case "elastic-net superposition of sinusoidal signals sparse correlated":
                            #     sub2model_parameters["number of features"]=ELASTIC_NET_SINUSOIDAL_SPARSE_CORRELATED_N_FEATURES
                            #     sub2model_parameters["number of informative features"]=ELASTIC_NET_SINUSOIDAL_SPARSE_CORRELATED_N_INFORMATIVE
                            #     sub2model_parameters["number of samples"]=ELASTIC_NET_SINUSOIDAL_SPARSE_CORRELATED_N_SAMPLES
                            #     for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                            #         N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                            #         SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, ELASTIC_NET_SINUSOIDAL_SPARSE_CORRELATED_N_FEATURES)
                            #         sub2model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS      
                            case "binary logistic regression":
                                sub3model_types=["binary logistic regression random 2-class classification"]
                                sub2model_parameters["C"]=BINARY_LOGISTIC_REGRESSION_C
                            # case "quadratic regression":
                            #     sub3model_types=["quadratic regression random quadratic regression"]
                            case "linear support vector classifier":
                                sub3model_types=["linear support vector classifier random 2-class classification"]
                                sub2model_parameters["C"]=LINEAR_SUPPORT_VECTOR_CLASSIFIER_C
                                sub2model_parameters["random state"]=LINEAR_SUPPORT_VECTOR_CLASSIFIER_RANDOM_STATE
                            case "linear support vector regressor":
                                sub3model_types=["linear support vector regressor random regression"]
                                sub2model_parameters["epsilon"]=LINEAR_SUPPORT_VECTOR_REGRESSOR_EPSILON
                                sub2model_parameters["C"]=LINEAR_SUPPORT_VECTOR_REGRESSOR_C
                                sub2model_parameters["random state"]=LINEAR_SUPPORT_VECTOR_REGRESSOR_RANDOM_REGRESSION_RANDOM_STATE
                            case "k nearest neighbors classifier":
                                sub3model_types=["k nearest neighbors classifier random 2-class classification"]
                                sub2model_parameters["number of neighbors"]=K_NEAREST_NEIGHBORS_CLASSIFICATION_N_NEIGHBORS
                            case "k nearest neighbors regressor":
                                sub3model_types=["k nearest neighbors regressor friedman #1"]
                                sub2model_parameters["number of neighbors"]=K_NEAREST_NEIGHBORS_REGRESSION_N_NEIGHBORS
                            case "decision tree classifier random 2-class classification":
                                sub2model_parameters["number of classes"]=DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLASSES
                                sub2model_parameters["number of clusters per class"]=DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_CLUSTERS_PER_CLASS
                                sub2model_parameters["number of features"]=DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES
                                sub2model_parameters["number of informative features"]=DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_INFORMATIVE
                                sub2model_parameters["number of redundant features"]=DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_REDUNDANT
                                sub2model_parameters["random state"]=DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_RANDOM_STATE
                                sub2model_parameters["number of samples"]=DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_SAMPLES
                                for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                                    N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                    SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, DECISION_TREE_CLASSIFIER_RANDOM_2_CLASS_CLASSIFICATION_N_FEATURES)
                                    sub2model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS
                            case "decision tree regressor friedman #1":
                                sub2model_parameters["number of features"]=DECISION_TREE_REGRESSOR_FRIEDMAN1_N_FEATURES
                                sub2model_parameters["number of samples"]=DECISION_TREE_REGRESSOR_FRIEDMAN1_N_SAMPLES
                                sub2model_parameters["random state"]=DECISION_TREE_REGRESSOR_FRIEDMAN1_RANDOM_STATE
                                for i,FEATURE_MATRIX_DDR in tqdm.tqdm(enumerate(FEATURE_MATRIX_DDRS), desc=f"Generating {sub2model_type} sets of features DDRs for each matrix of feature DDR: "):
                                    N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                    SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, DECISION_TREE_REGRESSOR_FRIEDMAN1_N_FEATURES)
                                    sub2model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS       
                            # case "random forest classifier":
                            #     sub3model_types=["random forest classifier random 2-class classification"]
                            #     sub2model_parameters["number of estimators"]=RANDOM_FOREST_CLASSIFIER_N_ESTIMATORS
                            #     sub2model_parameters["maximum number of features"]=RANDOM_FOREST_CLASSIFIER_MAX_FEATURES
                            #     sub2model_parameters["maximum depth"]=RANDOM_FOREST_CLASSIFIER_MAX_DEPTH
                            #     sub2model_parameters["minimum number of samples required to split an internal node"]=RANDOM_FOREST_CLASSIFIER_MIN_SAMPLES_SPLIT
                            #     sub2model_parameters["random state"]=RANDOM_FOREST_CLASSIFIER_RANDOM_STATE
                            # case "random forest regressor":
                            #     sub3model_types=["random forest regressor friedman #1"]
                            #     sub2model_parameters["number of estimators"]=RANDOM_FOREST_REGRESSOR_N_ESTIMATORS
                            #     sub2model_parameters["maximum number of features"]=RANDOM_FOREST_REGRESSOR_MAX_FEATURES
                            #     sub2model_parameters["maximum depth"]=RANDOM_FOREST_REGRESSOR_MAX_DEPTH
                            #     sub2model_parameters["minimum number of samples required to split an internal node"]=RANDOM_FOREST_REGRESSOR_MIN_SAMPLES_SPLIT
                            #     sub2model_parameters["random state"]=RANDOM_FOREST_REGRESSOR_RANDOM_STATE
                            case "multi-layer perceptron classifier":
                                sub3model_types=["multi-layer perceptron classifier random 2-class classification"]
                                sub2model_parameters["layer widths"]=MULTI_LAYER_PERCEPTRON_CLASSIFIER_LAYER_WIDTHS
                            case "multi-layer perceptron regressor":
                                sub3model_types=["multi-layer perceptron regressor friedman #1"]
                                sub2model_parameters["layer widths"]=MULTI_LAYER_PERCEPTRON_REGRESSOR_LAYER_WIDTHS
                            case "k means isotropic Gaussian blobs":
                                sub2model_parameters["number of centers"]=K_MEANS_ISOTROPIC_GAUSSIAN_BLOBS_CENTERS
                                sub2model_parameters["standard deviation of centers"]=K_MEANS_ISOTROPIC_GAUSSIAN_BLOBS_CLUSTER_STD
                                sub2model_parameters["number of features"]=K_MEANS_ISOTROPIC_GAUSSIAN_BLOBS_N_FEATURES
                                sub2model_parameters["number of samples"]=K_MEANS_ISOTROPIC_GAUSSIAN_BLOBS_N_SAMPLES
                                sub2model_parameters["random state"]=K_MEANS_ISOTROPIC_GAUSSIAN_BLOBS_RANDOM_STATE
                                for i,FEATURE_MATRIX_DDR in tqdm.tqdm(enumerate(FEATURE_MATRIX_DDRS), desc=f"Generating {sub2model_type} sets of feature DDRs for each matrix of feature DDR: "):
                                    N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                                    SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, K_MEANS_ISOTROPIC_GAUSSIAN_BLOBS_N_FEATURES)
                                    sub2model_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS
                        for sub3model_type in sub3model_types:
                            sub2model_parameters[sub3model_type]=Sub3model_Parameters(sub3model_type)
                        return sub2model_parameters
                    
                    submodel_parameters={}
                    sub2model_types=[]
                    match submodel_type:
                        case "ordinary least squares":
                            sub2model_types=["ordinary least squares random regression"]
                        # case "ridge regression and classification":
                        #     sub2model_types=["ridge regressor", "ridge classifier"]
                        # case "lasso":
                        #     sub2model_types=["lasso random regression sparse uncorrelated"]
                        #     submodel_parameters["alpha"]=LASSO_ALPHA
                        # case "elastic-net":
                        #     sub2model_types=["elastic-net superposition of sinusoidal signals sparse correlated"]
                        #     submodel_parameters["alpha"]=ELASTIC_NET_ALPHA
                        #     submodel_parameters["l1 ratio"]=ELASTIC_NET_L1_RATIO
                        case "logistic regression":
                            sub2model_types=["binary logistic regression"]
                        # case "polynomial regression":
                        #     sub2model_types=["quadratic regression"]
                        # case "kernel ridge regression friedman #1":
                        #     submodel_parameters["number of samples"]=KERNEL_RIDGE_REGRESSION_FRIEDMAN1_N_SAMPLES
                        #     submodel_parameters["number of features"]=KERNEL_RIDGE_REGRESSION_FRIEDMAN1_N_FEATURES
                        #     submodel_parameters["random state"]=KERNEL_RIDGE_REGRESSION_FRIEDMAN1_RANDOM_STATE
                        #     for i,FEATURE_MATRIX_DDR in enumerate(FEATURE_MATRIX_DDRS):
                        #         N_SETS_FEATURE_DDRS=NS_SETS_FEATURE_DDRS[i]
                        #         SETS_FEATURE_DDRS=Sets_Feature_DDRs(FEATURE_MATRIX_DDR, N_SETS_FEATURE_DDRS, KERNEL_RIDGE_REGRESSION_FRIEDMAN1_N_FEATURES)
                        #         submodel_parameters[f"matrix of features DDR == {FEATURE_MATRIX_DDR}"]=SETS_FEATURE_DDRS       
                        case "support vector classifier":
                            sub2model_types=["linear support vector classifier"]
                        case "support vector regressor":
                            sub2model_types=["linear support vector regressor"]
                        case "nearest neighbors classifier":
                            sub2model_types=["k nearest neighbors classifier"]
                        case "nearest neighbors regressor":
                            sub2model_types=["k nearest neighbors regressor"]
                        case "decision tree classifier":
                            sub2model_types=["decision tree classifier random 2-class classification"]
                            submodel_parameters["maximum depth"]=DECISION_TREE_CLASSIFIER_MAX_DEPTH
                            submodel_parameters["minimum number of samples required to be at a leaf"]=DECISION_TREE_CLASSIFIER_MIN_SAMPLES_LEAF
                            submodel_parameters["random state"]=DECISION_TREE_CLASSIFIER_RANDOM_STATE
                        case "decision tree regressor":
                            sub2model_types=["decision tree regressor friedman #1"]
                            submodel_parameters["maximum depth"]=DECISION_TREE_REGRESSOR_MAX_DEPTH
                            submodel_parameters["minimum number of samples required to be at a leaf"]=DECISION_TREE_REGRESSOR_MIN_SAMPLES_LEAF
                            submodel_parameters["random state"]=DECISION_TREE_REGRESSOR_RANDOM_STATE
                        # case "random forests":
                        #     sub2model_types=["random forest classifier", "random forest regressor"]
                        case "multi-layer perceptrons":
                            sub2model_types=["multi-layer perceptron classifier", "multi-layer perceptron regressor"]
                        case "k means":
                            sub2model_types=["k means isotropic Gaussian blobs"]
                            submodel_parameters["number of clusters"]=K_MEANS_N_CLUSTERS
                            submodel_parameters["random state"]=K_MEANS_RANDOM_STATE
                    for sub2model_type in sub2model_types:
                        sub2model_type_parameters=Sub2model_Parameters(sub2model_type)
                        submodel_parameters[sub2model_type]=sub2model_type_parameters
                    return submodel_parameters
            
                model_parameters={}
                submodel_types=[]
                match model_type:
                    case "linear models":
                        submodel_types=["ordinary least squares", "ridge regression and classification", "lasso", "elastic-net", "logistic regression", "polynomial regression"]
                    # case "kernel ridge regression":
                    #     model_parameters["alpha"]=KERNEL_RIDGE_REGRESSION_ALPHA
                    #     model_parameters["kernel"]=KERNEL_RIDGE_REGRESSION_KERNEL
                    #     model_parameters["gamma"]=KERNEL_RIDGE_REGRESSION_GAMMA
                    #     model_parameters["coef0"]=KERNEL_RIDGE_REGRESSION_COEF0
                    #     submodel_types=["kernel ridge regression friedman #1"]
                    case "support vector machines":
                        submodel_types=["support vector classifier", "support vector regressor"]
                    case "supervised nearest neighbors":
                        submodel_types=["nearest neighbors classifier", "nearest neighbors regressor"]
                    case "decision trees":
                        submodel_types=["decision tree classifier", "decision tree regressor"]
                    # case "ensembles":
                    #     submodel_types=["random forests"]
                    case "neural network models":
                        submodel_types=["multi-layer perceptrons"]
                    case "clustering":
                        submodel_types=["k means"]
                for submodel_type in submodel_types:
                    submodel_type_parameters=Submodel_Parameters(submodel_type)
                    model_parameters[submodel_type]=submodel_type_parameters
                return model_parameters
            
            learning_parameters={}
            model_types=[]
            if learning_type=="supervised learning":
                model_types=["linear models", "kernel ridge regression", "support vector machines", "supervised nearest neighbors", "decision trees", "ensembles", "neural network models"]
            elif learning_type=="unsupervised learning":
                model_types=["clustering"]
            for model_type in model_types:
                model_type_parameters=Model_Parameters(model_type)
                learning_parameters[model_type]=model_type_parameters             
            return learning_parameters
    
        parameters={}
        learning_types=["supervised learning", "unsupervised learning"]
        for learning_type in learning_types:
            learning_type_parameters=Learning_Parameters(learning_type)
            parameters[learning_type]=learning_type_parameters
        return parameters
    
    def Experiment(parameters): 
        """
        Runs the experiment and returns the results.
    
        Parameters
        ----------
        parameters : Dict
            The parameters for the experiment.
    
        Returns
        -------
        experiment : Dict
            The results of the experiment.
    
        """
    
        def Inputs(parameters):
            """
            Takes the parameters for the experiment and returns the inputs for the experiment
    
            Parameters
            ----------
            parameters : Dict
                The parameters of the experiment.
    
            Returns
            -------
            inputs : Dict
                The inputs for the experiment.
    
            """
            def make_sparse_correlated(n_samples=50, n_features=100, n_informative=10):
                """
                Generates a superposition of sinusoidal signals dataset with sparse, correlated design,
                where the number of samples is lower than the total number of features.
                Adapted from L1-based models for Sparse Signals by Arturo Amor <david-arturo.amor-quiroz@inria.fr>
                https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py
                
    
                Parameters
                ----------
                n_samples : int, optional
                    the number of samples. The default is 50.
                n_features : int, optional
                    the number of features. The default is 100.
                n_informative : int, optional
                    the number of informative features. The default is 10.
    
                Returns
                -------
                X : np.ndarray
                    Features matrix of frequencies.
                y : np.array
                    target values.
    
                """
                
                
                #The target y is a linear combination with alternating signs of sinusoidal signals.
                #Only the 10 lowest out of the 100 frequencies in X are used to generate y,
                #while the rest of the features are not informative.
                time_step = np.linspace(-2, 2, n_samples)
                freqs = 2 * np.pi * np.sort(rng.uniform(size=n_features)) / 0.01
                X = np.zeros((n_samples, n_features))
    
                for i in range(n_features):
                    X[:, i] = np.sin(freqs[i] * time_step)
    
                idx = np.arange(n_features)
                true_coef = (-1) ** idx * np.exp(-idx / 10)
                true_coef[n_informative:] = 0  # sparsify coef
                y = np.dot(X, true_coef)
    
                #A random phase is introduced using numpy.random.random_sample 
                for i in range(n_features):
                    X[:, i] = np.sin(freqs[i] * time_step + 2 * (rng.random() - 0.5))
                
                return X, y
    
            def make_quadratic_regression(
                n_samples=100,
                n_features=100,
                *,
                n_informative=10,
                n_targets=1,
                bias=0.0,
                effective_rank=None,
                tail_strength=0.5,
                noise=0.0,
                shuffle=True,
                coef=False,
                random_state=None,
            ):
                """Generate a random quadratic regression problem.
    
                The input set can either be well conditioned (by default) or have a low
                rank-fat tail singular profile. See :func:`make_low_rank_matrix` for
                more details.
    
                The output is generated by applying a (potentially biased) random quadratic
                regression model with `n_informative` nonzero regressors to the previously
                generated input and some gaussian centered noise with some adjustable
                scale.
                
                Adapted from make_regression by scikit-learn
                https://github.com/scikit-learn/scikit-learn/blob/2d8e03f4d/sklearn/datasets/_samples_generator.py#L535
    
                Parameters
                ----------
                n_samples : int, default=100
                    The number of samples.
    
                n_features : int, default=100
                    The number of features.
    
                n_informative : int, default=10
                    The number of informative features, i.e., the number of features used
                    to build the linear model used to generate the output.
    
                n_targets : int, default=1
                    The number of regression targets, i.e., the dimension of the y output
                    vector associated with a sample. By default, the output is a scalar.
    
                bias : float, default=0.0
                    The bias term in the underlying linear model.
    
                effective_rank : int, default=None
                    If not None:
                        The approximate number of singular vectors required to explain most
                        of the input data by linear combinations. Using this kind of
                        singular spectrum in the input allows the generator to reproduce
                        the correlations often observed in practice.
                    If None:
                        The input set is well conditioned, centered and gaussian with
                        unit variance.
    
                tail_strength : float, default=0.5
                    The relative importance of the fat noisy tail of the singular values
                    profile if `effective_rank` is not None. When a float, it should be
                    between 0 and 1.
    
                noise : float, default=0.0
                    The standard deviation of the gaussian noise applied to the output.
    
                shuffle : bool, default=True
                    Shuffle the samples and the features.
    
                coef : bool, default=False
                    If True, the coefficients of the underlying linear model are returned.
    
                random_state : int, RandomState instance or None, default=None
                    Determines random number generation for dataset creation. Pass an int
                    for reproducible output across multiple function calls.
                    See :term:`Glossary <random_state>`.
    
                Returns
                -------
                X_new : ndarray of shape (n_samples, NP)
                    The input samples. NP is the number of quadratic features generated from the combination of inputs.
    
                y : ndarray of shape (n_samples,) or (n_samples, n_targets)
                    The output values.
    
                coef : ndarray of shape (NP,) or (NP, n_targets)
                    The coefficients of the underlying quadratic model. It is returned only if
                    coef is True.
    
                """
                n_informative = min(n_features, n_informative)
                generator = sklearn_utils.check_random_state(random_state)
                
                quad=sklearn_preprocessing.PolynomialFeatures(degree=2)
                
                if effective_rank is None:
                    # Randomly generate a well conditioned input set
                    X = generator.standard_normal(size=(n_samples, n_features))
    
                else:
                    # Randomly generate a low rank, fat tail input set
                    X = sklearn_datasets.make_low_rank_matrix(
                        n_samples=n_samples,
                        n_features=n_features,
                        effective_rank=effective_rank,
                        tail_strength=tail_strength,
                        random_state=generator,
                    )
    
                X_new=quad.fit_transform(X)
                n_features_new=np.shape(X_new)[1]
                
                # Generate a ground truth model with only n_informative features being non
                # zeros (the other features are not correlated to y and should be ignored
                # by a sparsifying regularizers such as L1 or elastic net)
                ground_truth = np.zeros((n_features_new, n_targets))
                ground_truth[:n_informative, :] = 100 * generator.uniform(
                    size=(n_informative, n_targets)
                )
    
                y = np.dot(X_new, ground_truth) + bias
    
                # Add noise
                if noise > 0.0:
                    y += generator.normal(scale=noise, size=y.shape)
    
                # Randomly permute samples and features
                if shuffle:
                    from sklearn.utils import shuffle as util_shuffle  
                    X_new, y = util_shuffle(X_new, y, random_state=generator)
    
                    indices = np.arange(n_features_new)
                    generator.shuffle(indices)
                    X_new[:, :] = X_new[:, indices]
                    ground_truth = ground_truth[indices]
    
                y = np.squeeze(y)
    
                if coef:
                    return X_new, y, np.squeeze(ground_truth)
    
                else:
                    return X_new, y
            
            def Datasets(parameters):
                """
                Takes the parameters and returns the datasets
    
                Parameters
                ----------
                parameters : Dict
                    The parameters of the experiment.
    
                Returns
                -------
                datasets : Dict
                    The datasets for the experiment.
    
                """
                
                def Learning_Datasets(learning_parameters, learning_type):
                    """
                    Takes the parameters and learning type and returns the associated datasets
    
                    Parameters
                    ----------
                    learning_parameters: Dict
                        The parameters for the learning type for the experiment.
                    learning_type : string
                        Type of learning.
    
                    Returns
                    -------
                    learning_datasets: Dict
                        The datasets for type of learning learning_type.
    
                    """
                    
                    def Model_Datasets(model_parameters, model_type):
                        """
                        Takes the model parameters and the model type and returns the datasets for them.
    
                        Parameters
                        ----------
                        model_parameters: Dict
                            The parameters of the model
                        model_type : string
                            Type of model.
    
                        Returns
                        -------
                        model_datasets : Dict
                            The datasets of the model type.
    
                        """
    
                        def Submodel_Datasets(submodel_parameters, submodel_type):
                            """
                            Takes the submodel parameters and the type of submodel and return their datasets.
    
                            Parameters
                            ----------
                            submodel_parameters : Dict
                                The parameters of the submodel.
                            submodel_type : string
                                The type of submodel.
    
                            Returns
                            -------
                            submodel_datasets : Dict
                                The datasets for the submodel.
    
                            """
                            
                            def Sub2model_Datasets(sub2model_parameters, sub2model_type):
                                """
                                Takes the sub2model parameters and the type of sub2model and returns their datasets.
    
                                Parameters
                                ----------
                                sub2model_parameters : Dict
                                    The parameters of the sub2model.
                                sub2model_type : string
                                    Type of sub2model.
    
                                Returns
                                -------
                                sub2model_datasets : Dict
                                    The datasets for the sub2model.
    
                                """
                                
                                def Sub3model_Datasets(sub3model_parameters, sub3model_type):
                                    """
                                    Takes the sub3model parameters and the type of sub3model and returns their datasets.
    
                                    Parameters
                                    ----------
                                    sub3model_parameters : Dict
                                        The parameters of the sub3model.
                                    sub3model_type : string
                                        The type of sub3model.
    
                                    Returns
                                    -------
                                    sub3model_datasets : Dict
                                        The datasets for the sub3model.
    
                                    """
                                    
                                    sub3model_datasets={}
                                    match sub3model_type:
                                        # case "ridge regressor random regression":    
                                        #     n_features=sub3model_parameters["number of features"]
                                        #     n_informative=sub3model_parameters["number of informative features"]
                                        #     n_samples=sub3model_parameters["number of samples"]
                                        #     random_state=sub3model_parameters["random state"]
                                        #     sub3model_dataset=sklearn_datasets.make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=random_state)
                                        # case "ridge classifier random 2-class classification":
                                        #     n_classes=sub3model_parameters["number of classes"]
                                        #     n_clusters_per_class=sub3model_parameters["number of clusters per class"]
                                        #     n_features=sub3model_parameters["number of features"]
                                        #     n_informative=sub3model_parameters["number of informative features"]
                                        #     n_redundant=sub3model_parameters["number of redundant features"]
                                        #     n_samples=sub3model_parameters["number of samples"]
                                        #     random_state=sub3model_parameters["random state"]
                                        #     sub3model_dataset=sklearn_datasets.make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, random_state=random_state)
                                        case "binary logistic regression random 2-class classification":
                                            n_classes=sub3model_parameters["number of classes"]
                                            n_clusters_per_class=sub3model_parameters["number of clusters per class"]
                                            n_features=sub3model_parameters["number of features"]
                                            n_informative=sub3model_parameters["number of informative features"]
                                            n_redundant=sub3model_parameters["number of redundant features"]
                                            n_samples=sub3model_parameters["number of samples"]
                                            random_state=sub3model_parameters["random state"]
                                            sub3model_dataset=sklearn_datasets.make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, random_state=random_state)                                    
                                        # case "quadratic regression random quadratic regression":
                                        #     n_features=sub3model_parameters["number of features"]
                                        #     n_informative=sub3model_parameters["number of informative features"]
                                        #     n_samples=sub3model_parameters["number of samples"]
                                        #     random_state=sub3model_parameters["random state"]
                                        #     sub3model_dataset=make_quadratic_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=random_state)
                                        case "linear support vector classifier random 2-class classification":
                                            n_classes=sub3model_parameters["number of classes"]
                                            n_clusters_per_class=sub3model_parameters["number of clusters per class"]
                                            n_features=sub3model_parameters["number of features"]
                                            n_informative=sub3model_parameters["number of informative features"]
                                            n_redundant=sub3model_parameters["number of redundant features"]
                                            n_samples=sub3model_parameters["number of samples"]
                                            random_state=sub3model_parameters["random state"]
                                            sub3model_dataset=sklearn_datasets.make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, random_state=random_state)                                    
                                        case "linear support vector regressor random regression":
                                            n_features=sub3model_parameters["number of features"]
                                            n_informative=sub3model_parameters["number of informative features"]
                                            n_samples=sub3model_parameters["number of samples"]
                                            random_state=sub3model_parameters["random state"]
                                            sub3model_dataset=sklearn_datasets.make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=random_state)
                                        case "k nearest neighbors classifier random 2-class classification":
                                            n_classes=sub3model_parameters["number of classes"]
                                            n_clusters_per_class=sub3model_parameters["number of clusters per class"]
                                            n_features=sub3model_parameters["number of features"]
                                            n_informative=sub3model_parameters["number of informative features"]
                                            n_redundant=sub3model_parameters["number of redundant features"]
                                            n_samples=sub3model_parameters["number of samples"]
                                            random_state=sub3model_parameters["random state"]
                                            sub3model_dataset=sklearn_datasets.make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, random_state=random_state)
                                        case "k nearest neighbors regressor friedman #1":
                                            n_features=sub3model_parameters["number of features"]
                                            n_samples=sub3model_parameters["number of samples"]
                                            random_state=sub3model_parameters["random state"]
                                            sub3model_dataset=sklearn_datasets.make_friedman1(n_features=n_features, n_samples=n_samples, random_state=random_state)
                                        # case "random forest classifier random 2-class classification":
                                        #     n_classes=sub3model_parameters["number of classes"]
                                        #     n_clusters_per_class=sub3model_parameters["number of clusters per class"]
                                        #     n_features=sub3model_parameters["number of features"]
                                        #     n_informative=sub3model_parameters["number of informative features"]
                                        #     n_redundant=sub3model_parameters["number of redundant features"]
                                        #     n_samples=sub3model_parameters["number of samples"]
                                        #     random_state=sub3model_parameters["random state"]
                                        #     sub3model_dataset=sklearn_datasets.make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, random_state=random_state)
                                        # case "random forest regressor friedman #1":
                                        #     n_features=sub3model_parameters["number of features"]
                                        #     n_samples=sub3model_parameters["number of samples"]
                                        #     random_state=sub3model_parameters["random state"]
                                        #     sub3model_dataset=sklearn_datasets.make_friedman1(n_features=n_features, n_samples=n_samples, random_state=random_state)
                                        case "multi-layer perceptron classifier random 2-class classification":
                                            n_classes=sub3model_parameters["number of classes"]
                                            n_clusters_per_class=sub3model_parameters["number of clusters per class"]
                                            n_features=sub3model_parameters["number of features"]
                                            n_informative=sub3model_parameters["number of informative features"]
                                            n_redundant=sub3model_parameters["number of redundant features"]
                                            n_samples=sub3model_parameters["number of samples"]
                                            random_state=sub3model_parameters["random state"]
                                            sub3model_dataset=sklearn_datasets.make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, random_state=random_state)
                                        case "multi-layer perceptron regressor friedman #1":
                                            n_features=sub3model_parameters["number of features"]
                                            n_samples=sub3model_parameters["number of samples"]
                                            random_state=sub3model_parameters["random state"]
                                            sub3model_dataset=sklearn_datasets.make_friedman1(n_features=n_features, n_samples=n_samples, random_state=random_state)
                                    features=sub3model_dataset[0]
                                    targets=sub3model_dataset[1]
                                    features_train, features_test, targets_train, targets_test=sklearn_model_selection.train_test_split(features, targets, random_state=RANDOM_STATE)
                                    sub3model_datasets["training matrix of features"]=features_train
                                    sub3model_datasets["testing matrix of features"]=features_test
                                    sub3model_datasets["training targets"]=targets_train
                                    sub3model_datasets["testing targets"]=targets_test
                                    return sub3model_datasets
                                
                                sub2model_datasets={}
                                sub3model_types=[]
                                match sub2model_type:
                                    case "ordinary least squares random regression":
                                        n_features=sub2model_parameters["number of features"]
                                        n_informative=sub2model_parameters["number of informative features"]
                                        n_samples=sub2model_parameters["number of samples"]
                                        random_state=sub2model_parameters["random state"]
                                        sub2model_dataset=sklearn_datasets.make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=random_state)
                                        features=sub2model_dataset[0]
                                        targets=sub2model_dataset[1]
                                        features_train, features_test, targets_train, targets_test=sklearn_model_selection.train_test_split(features, targets, random_state=RANDOM_STATE)
                                        sub2model_datasets["training matrix of features"]=features_train
                                        sub2model_datasets["testing matrix of features"]=features_test
                                        sub2model_datasets["training targets"]=targets_train
                                        sub2model_datasets["testing targets"]=targets_test
                                    # case "ridge regressor":
                                    #     sub3model_types=["ridge regressor random regression"]
                                    # case "ridge classifier":
                                    #     sub3model_types=["ridge classifier random 2-class classification"]
                                    # case "lasso random regression sparse uncorrelated":
                                    #     n_features=sub2model_parameters["number of features"]
                                    #     n_samples=sub2model_parameters["number of samples"]
                                    #     random_state=sub2model_parameters["random state"]
                                    #     sub2model_dataset=sklearn_datasets.make_sparse_uncorrelated(n_features=n_features, n_samples=n_samples, random_state=random_state)
                                    #     features=sub2model_dataset[0]
                                    #     targets=sub2model_dataset[1]
                                    #     features_train, features_test, targets_train, targets_test=sklearn_model_selection.train_test_split(features, targets, random_state=RANDOM_STATE)
                                    #     sub2model_datasets["training matrix of features"]=features_train
                                    #     sub2model_datasets["testing matrix of features"]=features_test
                                    #     sub2model_datasets["training targets"]=targets_train
                                    #     sub2model_datasets["testing targets"]=targets_test
                                    # case "elastic-net superposition of sinusoidal signals sparse correlated":
                                    #     n_features=sub2model_parameters["number of features"]
                                    #     n_informative=sub2model_parameters["number of informative features"]
                                    #     n_samples=sub2model_parameters["number of samples"]
                                    #     sub2model_dataset=make_sparse_correlated(n_features=n_features, n_informative=n_informative, n_samples=n_samples)
                                    #     features=sub2model_dataset[0]
                                    #     targets=sub2model_dataset[1]
                                    #     features_train, features_test, targets_train, targets_test=sklearn_model_selection.train_test_split(features, targets, random_state=RANDOM_STATE)
                                    #     sub2model_datasets["training matrix of features"]=features_train
                                    #     sub2model_datasets["testing matrix of features"]=features_test
                                    #     sub2model_datasets["training targets"]=targets_train
                                    #     sub2model_datasets["testing targets"]=targets_test
                                    case "binary logistic regression":
                                        sub3model_types=["binary logistic regression random 2-class classification"]
                                    # case "quadratic regression":
                                    #     sub3model_types=["quadratic regression random quadratic regression"]
                                    case "linear support vector classifier":
                                        sub3model_types=["linear support vector classifier random 2-class classification"]
                                    case "linear support vector regressor":
                                        sub3model_types=["linear support vector regressor random regression"]
                                    case "k nearest neighbors classifier":
                                        sub3model_types=["k nearest neighbors classifier random 2-class classification"]
                                    case "k nearest neighbors regressor":
                                        sub3model_types=["k nearest neighbors regressor friedman #1"]
                                    case "decision tree classifier random 2-class classification":
                                        n_classes=sub2model_parameters["number of classes"]
                                        n_clusters_per_class=sub2model_parameters["number of clusters per class"]
                                        n_features=sub2model_parameters["number of features"]
                                        n_informative=sub2model_parameters["number of informative features"]
                                        n_redundant=sub2model_parameters["number of redundant features"]
                                        n_samples=sub2model_parameters["number of samples"]
                                        random_state=sub2model_parameters["random state"]
                                        sub2model_dataset=sklearn_datasets.make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, random_state=random_state)
                                        features=sub2model_dataset[0]
                                        targets=sub2model_dataset[1]
                                        features_train, features_test, targets_train, targets_test=sklearn_model_selection.train_test_split(features, targets, random_state=RANDOM_STATE)
                                        sub2model_datasets["training matrix of features"]=features_train
                                        sub2model_datasets["testing matrix of features"]=features_test
                                        sub2model_datasets["training targets"]=targets_train
                                        sub2model_datasets["testing targets"]=targets_test
                                    case "decision tree regressor friedman #1":
                                        n_features=sub2model_parameters["number of features"]
                                        n_samples=sub2model_parameters["number of samples"]
                                        random_state=sub2model_parameters["random state"]
                                        sub2model_dataset=sklearn_datasets.make_friedman1(n_features=n_features, n_samples=n_samples, random_state=random_state)
                                        features=sub2model_dataset[0]
                                        targets=sub2model_dataset[1]
                                        features_train, features_test, targets_train, targets_test=sklearn_model_selection.train_test_split(features, targets, random_state=RANDOM_STATE)
                                        sub2model_datasets["training matrix of features"]=features_train
                                        sub2model_datasets["testing matrix of features"]=features_test
                                        sub2model_datasets["training targets"]=targets_train
                                        sub2model_datasets["testing targets"]=targets_test
                                    # case "random forest classifier":
                                    #     sub3model_types=["random forest classifier random 2-class classification"]
                                    # case "random forest regressor":
                                    #     sub3model_types=["random forest regressor friedman #1"]
                                    case "multi-layer perceptron classifier":
                                        sub3model_types=["multi-layer perceptron classifier random 2-class classification"]
                                    case "multi-layer perceptron regressor":
                                        sub3model_types=["multi-layer perceptron regressor friedman #1"]
                                    case "k means isotropic Gaussian blobs":
                                        centers=sub2model_parameters["number of centers"]
                                        cluster_std=sub2model_parameters["standard deviation of centers"]
                                        n_features=sub2model_parameters["number of features"]
                                        n_samples=sub2model_parameters["number of samples"]
                                        random_state=sub2model_parameters["random state"]
                                        sub2model_dataset=sklearn_datasets.make_blobs(centers=centers, cluster_std=cluster_std, n_features=n_features, n_samples=n_samples, random_state=random_state)
                                        features=sub2model_dataset[0]
                                        targets=sub2model_dataset[1]
                                        sub2model_datasets["matrix of features"]=features
                                        sub2model_datasets["targets"]=targets
                                for sub3model_type in sub3model_types:
                                    sub3model_parameters=sub2model_parameters[sub3model_type]
                                    sub3model_datasets=Sub3model_Datasets(sub3model_parameters,sub3model_type)
                                    sub2model_datasets[sub3model_type]=sub3model_datasets
                                return sub2model_datasets
                            
                            submodel_datasets={}
                            sub2model_types=[]
                            match submodel_type:
                                case "ordinary least squares":
                                    sub2model_types=["ordinary least squares random regression"]
                                # case "ridge regression and classification":
                                #     sub2model_types=["ridge regressor", "ridge classifier"]
                                # case "lasso":
                                #     sub2model_types=["lasso random regression sparse uncorrelated"]
                                # case "elastic-net":
                                #     sub2model_types=["elastic-net superposition of sinusoidal signals sparse correlated"]
                                case "logistic regression":
                                    sub2model_types=["binary logistic regression"]
                                # case "polynomial regression":
                                #     sub2model_types=["quadratic regression"]
                                # case "kernel ridge regression friedman #1":
                                #     n_features=submodel_parameters["number of features"]
                                #     n_samples=submodel_parameters["number of samples"]
                                #     random_state=submodel_parameters["random state"]
                                #     submodel_dataset=sklearn_datasets.make_friedman1(n_features=n_features, n_samples=n_samples, random_state=random_state)
                                #     features=submodel_dataset[0]
                                #     targets=submodel_dataset[1]
                                #     features_train, features_test, targets_train, targets_test=sklearn_model_selection.train_test_split(features, targets, random_state=RANDOM_STATE)
                                #     submodel_datasets["training matrix of features"]=features_train
                                #     submodel_datasets["testing matrix of features"]=features_test
                                #     submodel_datasets["training targets"]=targets_train
                                #     submodel_datasets["testing targets"]=targets_test
                                case "support vector classifier":
                                    sub2model_types=["linear support vector classifier"]
                                case "support vector regressor":
                                    sub2model_types=["linear support vector regressor"]
                                case "nearest neighbors classifier":
                                    sub2model_types=["k nearest neighbors classifier"]
                                case "nearest neighbors regressor":
                                    sub2model_types=["k nearest neighbors regressor"]
                                case "decision tree classifier":
                                    sub2model_types=["decision tree classifier random 2-class classification"]
                                case "decision tree regressor":
                                    sub2model_types=["decision tree regressor friedman #1"]
                                # case "random forests":
                                #     sub2model_types=["random forest classifier", "random forest regressor"]
                                case "multi-layer perceptrons":
                                    sub2model_types=["multi-layer perceptron classifier", "multi-layer perceptron regressor"]
                                case "k means":
                                    sub2model_types=["k means isotropic Gaussian blobs"]
                            for sub2model_type in sub2model_types:
                                sub2model_parameters=submodel_parameters[sub2model_type]
                                sub2model_datasets=Sub2model_Datasets(sub2model_parameters,sub2model_type)
                                submodel_datasets[sub2model_type]=sub2model_datasets
                            return submodel_datasets
                    
                        model_datasets={}
                        submodel_types=[]
                        match model_type:
                            case "linear models":
                                submodel_types=["ordinary least squares", "ridge regression and classification", "lasso", "elastic-net", "logistic regression", "polynomial regression"]
                            # case "kernel ridge regression":
                            #     submodel_types=["kernel ridge regression friedman #1"]
                            case "support vector machines":
                                submodel_types=["support vector classifier", "support vector regressor"]
                            case "supervised nearest neighbors":
                                submodel_types=["nearest neighbors classifier", "nearest neighbors regressor"]
                            case "decision trees":
                                submodel_types=["decision tree classifier", "decision tree regressor"]
                            # case "ensembles":
                            #     submodel_types=["random forests"]
                            case "neural network models":
                                submodel_types=["multi-layer perceptrons"]
                            case "clustering":
                                submodel_types=["k means"]
                        for submodel_type in submodel_types:
                            submodel_parameters=model_parameters[submodel_type]
                            submodel_datasets=Submodel_Datasets(submodel_parameters,submodel_type)
                            model_datasets[submodel_type]=submodel_datasets
                        return model_datasets
                    
                    learning_datasets={}
                    model_types=[]
                    if learning_type=="supervised learning":
                        model_types=["linear models", "kernel ridge regression", "support vector machines", "supervised nearest neighbors", "decision trees", "ensembles", "neural network models"]
                    elif learning_type=="unsupervised learning":
                        model_types=["clustering"]
                    for model_type in model_types:
                        model_parameters=learning_parameters[model_type]
                        model_datasets=Model_Datasets(model_parameters,model_type)
                        learning_datasets[model_type]=model_datasets
                    return learning_datasets
                
                datasets={}
                learning_types=["supervised learning", "unsupervised learning"]
                for learning_type in learning_types:
                    learning_parameters=parameters[learning_type]            
                    learning_datasets=Learning_Datasets(learning_parameters,learning_type)
                    datasets[learning_type]=learning_datasets
                return datasets
            
            class Set_Feature_DDRs_Invariant_Standard_Scaler(sklearn_base.OneToOneFeatureMixin, sklearn_base.TransformerMixin, sklearn_base.BaseEstimator):
                """Standardize noisy features while preserving set of features DDRs by, for each clean feature,
                finding the mean and standard deviation of the clean feature,
                calculating scaling and shifting parameters,
                scaling and shifting the clean feature according to these parameters,
                calculating the desired standard deviation of the noise component of the standardized feature,
                generating noise with this variance, and adding the noise to the clean feature.
                
                The scaling and shifting parameters `alpha` and `beta` and the desired standard
                deviation of the noise `sigma` are calculated as:
                    
                    alpha = r^(1/2) / s
                    
                    beta = - u / s * r^(1/2)
      
                    sigma = (1 - r)^(1/2)
                    
                where `r` is the desired feature DDR of the feature samples, `s` is the standard deviation of the feature samples,
                and `u` is the sample mean of the feature samples.
            
                Centering and scaling happen independently on each feature by computing
                the relevant statistics on the samples in the training set. The scaling and
                shifting parameters, as well as the standard deviation of the noise
                for the standardized noisy feature, are then stored to be used on later data using
                :meth:`transform`.
            
                Standardization of a dataset is a common requirement for many
                machine learning estimators: they might behave badly if the
                individual features do not more or less look like standard normally
                distributed data (e.g. Gaussian with 0 mean and unit variance).
            
                For instance many elements used in the objective function of
                a learning algorithm (such as the RBF kernel of Support Vector
                Machines or the L1 and L2 regularizers of linear models) assume that
                all features are centered around 0 and have variance in the same
                order. If a feature has a variance that is orders of magnitude larger
                than others, it might dominate the objective function and make the
                estimator unable to learn from other features correctly as expected.
            
                However, ordinary standardization of a dataset also changes the DDR of the dataset. 
                For the purposes of this experiment, a DDR-invariant approach to standardization
                is necessary so that, when the evaluation metric of the model vs. features DDR is plotted,
                the values of features DDR, initialized at the beginning of the program,
                are the same as the values of features DDR of the datasets used to train the model.
                        
                `Set_Feature_DDRs_Invariant_Standard_Scaler` is sensitive to outliers, and the features may scale
                differently from each other in the presence of outliers. For an example
                visualization, refer to :ref:`Compare StandardScaler with other scalers
                <plot_all_scaling_standard_scaler_section>`.
            
                Parameters
                ----------
                copy : bool, default=True
                    If False, try to avoid a copy and do inplace scaling instead.
                    This is not guaranteed to always work inplace; e.g. if the data is
                    not a NumPy array or scipy.sparse CSR matrix, a copy may still be
                    returned.
            
                Attributes
                ----------
                scale_ : ndarray of shape (n_features,) or None
                    Per feature relative scaling of the data to achieve zero mean and unit
                    variance. Generally this is calculated using `np.sqrt(var_)`. If a
                    variance is zero, we can't achieve unit variance, and the data is left
                    as-is, giving a scaling factor of 1. 
                    
                       *scale_*
            
                alpha_ : ndarray of shape (n_features,) or None
                    Scaling factor for each feature in training set..
            
                beta_ : ndarray of shape (n_features,) or None
                    The shifting term for each feature in the training set.
            
                n_features_in_ : int
                    Number of features seen during :term:`fit`.
            
                    .. versionadded:: 0.24
            
                feature_names_in_ : ndarray of shape (`n_features_in_`,)
                    Names of features seen during :term:`fit`. Defined only when `X`
                    has feature names that are all strings.
            
                    .. versionadded:: 1.0
            
                n_samples_seen_ : int or ndarray of shape (n_features,)
                    The number of samples processed by the estimator for each feature.
                    If there are no missing samples, the ``n_samples_seen`` will be an
                    integer, otherwise it will be an array of dtype int. If
                    `sample_weights` are used it will be a float (if no missing data)
                    or an array of dtype float that sums the weights seen so far.
                    Will be reset on new calls to fit, but increments across
                    ``partial_fit`` calls.
            
                Notes
                -----
                NaNs are treated as missing values: disregarded in fit, and maintained in
                transform.
            
                We use a biased estimator for the standard deviation, equivalent to
                `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
                affect model performance.
            
                """
            
                _parameter_constraints: dict = {
                    "copy": ["boolean"],
                }
            
                def __init__(self, set_feature_ddrs, copy=True):
                    self.copy = copy
                    self.set_feature_ddrs=set_feature_ddrs
                    self.sigma=np.sqrt(1-set_feature_ddrs)
                    
                    
                def _reset(self):
                    """Reset internal data-dependent state of the scaler, if necessary.
            
                    __init__ parameters are not touched.
                    """
                    # Checking one attribute is enough, because they are all set together
                    # in partial_fit
                    if hasattr(self, "scale_"):
                        del self.scale_
                        del self.n_samples_seen_
                        del self.alpha_
                        del self.beta_
            
                def fit(self, X, y=None, sample_weight=None):
                    """Compute the scaling and shifting parameters to be used for later transformation.
            
                    Parameters
                    ----------
                    X : {array-like, sparse matrix} of shape (n_samples, n_features)
                        The data used to compute the scaling and shifting parameters deviation
                        used for later scaling along the features axis.
            
                    y : None
                        Ignored.
            
                    sample_weight : array-like of shape (n_samples,), default=None
                        Individual weights for each sample.
            
                    Returns
                    -------
                    self : object
                        Fitted set of feature DDRs-invariant scaler.
                    """
                    # Reset internal state before fitting
                    self._reset()
                    return self.partial_fit(X, y, sample_weight)
            
                @sklearn_base._fit_context(prefer_skip_nested_validation=True)
                def partial_fit(self, X, y=None, sample_weight=None):
                    """Online computation of scaling factor and shifting term on X for later transformation.
            
                    All of X is processed as a single batch. This is intended for cases
                    when :meth:`fit` is not feasible due to very large number of
                    `n_samples` or because X is read from a continuous stream.
            
                    The algorithm for incremental mean and std is given in Equation 1.5a,b
                    in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
                    for computing the sample variance: Analysis and recommendations."
                    The American Statistician 37.3 (1983): 242-247:
            
                    Parameters
                    ----------
                    X : {array-like, sparse matrix} of shape (n_samples, n_features)
                        The data used to compute scaling factor and shifting term
                        used for later transformation along the features axis.
            
                    y : None
                        Ignored.
            
                    sample_weight : array-like of shape (n_samples,), default=None
                        Individual weights for each sample.
            
                    Returns
                    -------
                    self : object
                        Fitted set of feature DDRs-invariant scaler.
                    """
                    first_call = not hasattr(self, "n_samples_seen_")
                    X = self._validate_data(
                        X,
                        accept_sparse=("csr", "csc"),
                        dtype=sklearn_utils.validation.FLOAT_DTYPES,
                        force_all_finite="allow-nan",
                        reset=first_call,
                    )
                    n_features = X.shape[1]
            
                    if sample_weight is not None:
                        sample_weight = sklearn_utils.validation._check_sample_weight(sample_weight, X, dtype=X.dtype)
            
                    # Even in the case of `with_mean=False`, we update the mean anyway
                    # This is needed for the incremental computation of the var
                    # See incr_mean_variance_axis and _incremental_mean_variance_axis
            
                    # if n_samples_seen_ is an integer (i.e. no missing values), we need to
                    # transform it to a NumPy array of shape (n_features,) required by
                    # incr_mean_variance_axis and _incremental_variance_axis
                    dtype = np.int64 if sample_weight is None else X.dtype
                    if not hasattr(self, "n_samples_seen_"):
                        self.n_samples_seen_ = np.zeros(n_features, dtype=dtype)
                    elif np.size(self.n_samples_seen_) == 1:
                        self.n_samples_seen_ = np.repeat(self.n_samples_seen_, X.shape[1])
                        self.n_samples_seen_ = self.n_samples_seen_.astype(dtype, copy=False)
            
                    if scipy.sparse.issparse(X):
                        
                        raise ValueError(
                            "Cannot center sparse matrices: pass `with_mean=False` "
                            "instead. See docstring for motivation and alternatives."
                        )
                        sparse_constructor = (
                            scipy.sparse.csr_matrix if X.format == "csr" else scipy.sparse.csc_matrix
                        )
            
                        
                        # First pass
                        if not hasattr(self, "scale_"):
                            self.mean_, self.var_, self.n_samples_seen_ = scipy.utils.sparsefuncs.mean_variance_axis(
                                X, axis=0, weights=sample_weight, return_sum_weights=True
                            )
                        # Next passes
                        else:
                            (
                                self.mean_,
                                self.var_,
                                self.n_samples_seen_,
                            ) = scipy.utils.sparsefuncs.incr_mean_variance_axis(
                                X,
                                axis=0,
                                last_mean=self.mean_,
                                last_var=self.var_,
                                last_n=self.n_samples_seen_,
                                weights=sample_weight,
                            )
                        # We force the mean and variance to float64 for large arrays
                        # See https://github.com/scikit-learn/scikit-learn/pull/12338
                        self.mean_ = self.mean_.astype(np.float64, copy=False)
                        self.var_ = self.var_.astype(np.float64, copy=False)
                    else:
                        # First pass
                        if not hasattr(self, "scale_"):
                            self.mean_ = 0.0
                            
                            self.var_ = 0.0
                            
                                
                        self.mean_, self.var_, self.n_samples_seen_ = sklearn_utils.extmath._incremental_mean_and_var(
                            X,
                            self.mean_,
                            self.var_,
                            self.n_samples_seen_,
                            sample_weight=sample_weight,
                        )
                        
                    # for backward-compatibility, reduce n_samples_seen_ to an integer
                    # if the number of samples is the same for each feature (i.e. no
                    # missing values)
                    if np.ptp(self.n_samples_seen_) == 0:
                        self.n_samples_seen_ = self.n_samples_seen_[0]
            
                    # Extract the list of near constant features on the raw variances,
                    # before taking the square root.
                    def _is_constant_feature(var, mean, n_samples):
                        """Detect if a feature is indistinguishable from a constant feature.
                    
                        The detection is based on its computed variance and on the theoretical
                        error bounds of the '2 pass algorithm' for variance computation.
                    
                        See "Algorithms for computing the sample variance: analysis and
                        recommendations", by Chan, Golub, and LeVeque.
                        """
                        # In scikit-learn, variance is always computed using float64 accumulators.
                        eps = np.finfo(np.float64).eps
                    
                        upper_bound = n_samples * eps * var + (n_samples * mean * eps) ** 2
                        return var <= upper_bound
    
                    constant_mask = _is_constant_feature(
                        self.var_, self.mean_, self.n_samples_seen_
                    )
                    
                    def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
                        """Set scales of near constant features to 1.
                    
                        The goal is to avoid division by very small or zero values.
                    
                        Near constant features are detected automatically by identifying
                        scales close to machine precision unless they are precomputed by
                        the caller and passed with the `constant_mask` kwarg.
                    
                        Typically for standard scaling, the scales are the standard
                        deviation while near constant features are better detected on the
                        computed variances which are closer to machine precision by
                        construction.
                        """
                        # if we are fitting on 1D arrays, scale might be a scalar
                        if np.isscalar(scale):
                            if scale == 0.0:
                                scale = 1.0
                            return scale
                        elif isinstance(scale, np.ndarray):
                            if constant_mask is None:
                                # Detect near constant values to avoid dividing by a very small
                                # value that could lead to surprising results and numerical
                                # stability issues.
                                constant_mask = scale < 10 * np.finfo(scale.dtype).eps
                    
                            if copy:
                                # New array to avoid side-effects
                                scale = scale.copy()
                            scale[constant_mask] = 1.0
                            return scale
                        
                    self.scale_ = _handle_zeros_in_scale(
                        np.sqrt(self.var_), copy=False, constant_mask=constant_mask
                    )
    
                    self.alpha_=np.ones(n_features)
                    self.beta_=-1*self.mean_                
                    for i,var in enumerate(self.var_):
                        if var!=0:
                            feature_ddr=self.set_feature_ddrs[i]
                            mean=self.mean_[i]
                            alpha=(feature_ddr/var)**(1/2)
                            beta=-mean*(feature_ddr/var)**(1/2)
                            self.alpha_[i]=alpha
                            self.beta_[i]=beta
            
                    return self
            
                def transform(self, X, copy=None):
                    """Perform set of feature DDRs-invariant standardization by centering the clean feature,
                    scaling the clean feature, generating noise, and adding it to the clean feature.
            
                    Parameters
                    ----------
                    X : {array-like, sparse matrix of shape (n_samples, n_features)
                        The data used to scale along the features axis.
                    copy : bool, default=None
                        Copy the input X or not.
            
                    Returns
                    -------
                    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
                        Transformed array.
                    """
                    sklearn_utils.validation.check_is_fitted(self)
            
                    copy = copy if copy is not None else self.copy
                    X = self._validate_data(
                        X,
                        reset=False,
                        accept_sparse="csr",
                        copy=copy,
                        dtype=sklearn_utils.validation.FLOAT_DTYPES,
                        force_all_finite="allow-nan",
                    )
            
                    if scipy.sparse.issparse(X):
                        raise ValueError(
                            "Cannot center sparse matrices: pass `with_mean=False` "
                            "instead. See docstring for motivation and alternatives."
                        )
                        if self.scale_ is not None:
                            sklearn_utils.sparsefuncs.inplace_column_scale(X, 1 / self.scale_)
                    else:
                        X =np.multiply(X,self.alpha_)
                        X = np.add(X,self.beta_)
                        noise=rng.normal(0, self.sigma, X.shape)
                        X = np.add(X,noise)
                    return X
            
                def inverse_transform(self, X, copy=None):
                    """Scale back the data to the original representation.
            
                    Parameters
                    ----------
                    X : {array-like, sparse matrix} of shape (n_samples, n_features)
                        The data used to scale along the features axis.
                    copy : bool, default=None
                        Copy the input X or not.
            
                    Returns
                    -------
                    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
                        Transformed array.
                    """
                    sklearn_utils.validation.check_is_fitted(self)
            
                    copy = copy if copy is not None else self.copy
                    X = sklearn_utils.check_array(
                        X,
                        accept_sparse="csr",
                        copy=copy,
                        dtype=sklearn_utils.validation.FLOAT_DTYPES,
                        force_all_finite="allow-nan",
                    )
            
                    if scipy.sparse.issparse(X):
                        raise ValueError(
                            "Cannot uncenter sparse matrices: pass `with_mean=False` "
                            "instead See docstring for motivation and alternatives."
                        )
                    else:
                        X = np.subtract(X,self.beta_)
                        X = np.divide(X,self.alpha_)
                    return X
            
                def _more_tags(self):
                    return {"allow_nan": True, "preserves_dtype": [np.float64, np.float32]}
                
            class MultiLayerPerceptron(torch.nn.Module):
                """
                Class for Multi-Layer Perceptron. Inherits from torch.nn.Module class.
                Adapted from PyTorch-Multilayer_Perceptron by tmiethlinger:
                https://github.com/tmiethlinger/PyTorch-Multilayer_Perceptron
                """
                
                def __init__(self, layer_widths, activation_function):
                    super(MultiLayerPerceptron, self).__init__()
    
                    self.input_size = layer_widths[0]
                    self.output_size = layer_widths[-1]
    
                    self.layer_widths = layer_widths
    
                    self.activation_function = activation_function
    
                    self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(w[0], w[1]) for w in zip(self.layer_widths[:-1], self.layer_widths[1:])])
    
                    self.activation_layers = torch.nn.ModuleList([self.activation_function for a in range(len(self.layer_widths) - 1)])
    
                    self.double()
                    
                def forward(self, x):
    
                    # State s is input data x
                    s = x
    
                    # Iterate layers
                    for linear, activation in zip(self.linear_layers[:-1], self.activation_layers):
                        s = linear(s)
                        s=activation(s)
                    # Output y
                    y = self.linear_layers[-1](s)
                    return y
             
            
            # RBF Layer
            
            class RBF(torch.nn.Module):
                """
                Transforms incoming data using a given radial basis function:
                u_{i} = rbf(||x - c_{i}|| / s_{i})
            
                Arguments:
                    in_features: size of each input sample
                    out_features: size of each output sample
            
                Shape:
                    - Input: (N, in_features) where N is an arbitrary batch size
                    - Output: (N, out_features) where N is an arbitrary batch size
            
                Attributes:
                    centres: the learnable centres of shape (out_features, in_features).
                        The values are initialised from a standard normal distribution.
                        Normalising inputs to have mean 0 and standard deviation 1 is
                        recommended.
                    
                    log_sigmas: logarithm of the learnable scaling factors of shape (out_features).
                    
                    basis_func: the radial basis function used to transform the scaled
                        distances.
                """ 
           
                def __init__(self, in_features, out_features, basis_func):
                    super(RBF, self).__init__()
                    self.in_features = in_features
                    self.out_features = out_features
                    self.centres = torch.nn.Parameter(torch.Tensor(out_features, in_features))
                    self.log_sigmas = torch.nn.Parameter(torch.Tensor(out_features))
                    self.basis_func = basis_func
                    self.reset_parameters()
            
                def reset_parameters(self):
                    torch.nn.init.normal_(self.centres, 0, 1)
                    torch.nn.init.constant_(self.log_sigmas, 0)
            
                def forward(self, input):
                    size = (input.size(0), self.out_features, self.in_features)
                    x = input.unsqueeze(1).expand(size)
                    c = self.centres.unsqueeze(0).expand(size)
                    distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
                    return self.basis_func(distances)
            
               
                
                def basis_func_dict():
                    """
                    A helper function that returns a dictionary containing each RBF
                    """
               
                    # RBFs
                    
                    def gaussian(alpha):
                        phi = torch.exp(-1*alpha.pow(2))
                        return phi
                    
                    def linear(alpha):
                        phi = alpha
                        return phi
                    
                    def quadratic(alpha):
                        phi = alpha.pow(2)
                        return phi
                    
                    def inverse_quadratic(alpha):
                        phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
                        return phi
                    
                    def multiquadric(alpha):
                        phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
                        return phi
                    
                    def inverse_multiquadric(alpha):
                        phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
                        return phi
                    
                    def spline(alpha):
                        phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
                        return phi
                    
                    def poisson_one(alpha):
                        phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
                        return phi
                    
                    def poisson_two(alpha):
                        phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
                        * alpha * torch.exp(-alpha)
                        return phi
                    
                    def matern32(alpha):
                        phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
                        return phi
                    
                    def matern52(alpha):
                        phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3) \
                        * alpha.pow(2))*torch.exp(-5**0.5*alpha)
                        return phi     
               
                    bases = {'gaussian': gaussian,
                             'linear': linear,
                             'quadratic': quadratic,
                             'inverse quadratic': inverse_quadratic,
                             'multiquadric': multiquadric,
                             'inverse multiquadric': inverse_multiquadric,
                             'spline': spline,
                             'poisson one': poisson_one,
                             'poisson two': poisson_two,
                             'matern32': matern32,
                             'matern52': matern52}
                    return bases
            
            class RadialBasisFunctionNetwork(torch.nn.Module):
        
                def __init__(self, layer_widths, layer_centres, basis_func):
                    super(RadialBasisFunctionNetwork, self).__init__()
                    self.rbf_layers = torch.nn.ModuleList()
                    self.linear_layers = torch.nn.ModuleList()
                    for i in range(len(layer_widths) - 1):
                        self.rbf_layers.append(RBF(layer_widths[i], layer_centres[i], basis_func))
                        self.linear_layers.append(torch.nn.Linear(layer_centres[i], layer_widths[i+1]))
                
                def forward(self, x):
                    out = x
                    for i in range(len(self.rbf_layers)):
                        out = self.rbf_layers[i](out)
                        out = self.linear_layers[i](out)
                    return out
            
           
    
            def Pipes(parameters):
                """
                Returns the untrained pipes for the experiment
    
                Parameters
                ----------
                parameters : Dict
                    Parameters for the experiment.
    
                Returns
                -------
                pipes: Dict
                    Untrained pipes for the experiment.
    
                """
                  
                def Learning_Pipes(learning_parameters, learning_type):
                    """
                    Takes learning parameters and a learning type and returns the associated untrained pipes
    
                    Parameters
                    ----------
                    learning_parameters : Dict
                        Parameters of the learning type.
                    learning_type : string
                        Type of learning.
    
                    Returns
                    -------
                    learning_pipes: Dict
                        Untrained pipes for type of learning learning_type.
    
                    """
                    
                    def Model_Pipes(model_parameters, model_type):
                        """
                        Takes model parameters and a model type and returns the untrained pipes for them.
    
                        Parameters
                        ----------
                        model_parameters : Dict
                            Parameters of the model type.
                        model_type : string
                            Type of model.
    
                        Returns
                        -------
                        model_pipes : Dict
                            Untrained pipes of the model type.
    
                        """
    
                        def Submodel_Pipes(submodel_parameters, submodel_type):
                            """
                            Takes submodel parameters and a type of submodel and returns their untrained pipes.
    
                            Parameters
                            ----------
                            submodel_parameters : Dict
                                Parameters of the submodel type.
                            submodel_type : string
                                The type of submodel.
    
                            Returns
                            -------
                            submodel_pipes : Dict
                                Untrained pipes for the submodel.
    
                            """
                            
                            def Sub2model_Pipes(sub2model_parameters, sub2model_type):
                                """
                                Takes sub2model parameters and a type of sub2model and returns their untrained pipes.
    
                                Parameters
                                ----------
                                sub2model_parameters : Dict
                                    Parameters of the sub2model type.
                                sub2model_type : string
                                    Type of sub2model.
    
                                Returns
                                -------
                                sub2model_pipes : Dict
                                    Untrained pipes for the sub2model.
    
                                """
                                
                                sub2model_pipes={}
                                sub3model_types=[]
                                match sub2model_type:
                                    # case "ridge regressor":
                                    #     sub3model_types=["ridge regressor random regression"]
                                    #     alpha=sub2model_parameters["alpha"]
                                    #     random_state=sub2model_parameters["random state"]
                                    #     sub2model_model=sklearn_linear_model.Ridge(alpha=alpha, random_state=random_state)
                                    # case "ridge classifier":
                                    #     sub3model_types=["ridge classifier random 2-class classification"]
                                    #     alpha=sub2model_parameters["alpha"]
                                    #     random_state=sub2model_parameters["random state"]
                                    #     sub2model_model=sklearn_linear_model.RidgeClassifier(alpha=alpha, random_state=random_state)
                                    case "binary logistic regression":
                                        sub3model_type="binary logistic regression random 2-class classification"
                                        C=sub2model_parameters["C"]
                                        sub2model_model=sklearn_linear_model.LogisticRegression(C=C)
                                        sub3model_parameters=sub2model_parameters[sub3model_type]
                                        sub3model_pipes={}
                                        for key in tqdm.tqdm(sub3model_parameters.keys(), desc=f"Generating pipes for {sub3model_type} matrix of features DDRs: \n"):
                                            if key not in ["number of features", "number of samples", "number of informative features", "number of redundant features", "number of classes", "number of clusters per class", "random state"]:
                                                sets_feature_ddrs=sub3model_parameters[key]
                                                sub4model_pipes={}
                                                for set_feature_ddrs in tqdm.tqdm(sets_feature_ddrs, desc=f"Generating pipes for {key} sets feature DDRs: \n"):
                                                    scaler= Set_Feature_DDRs_Invariant_Standard_Scaler(set_feature_ddrs)
                                                    sub4model_pipe=sklearn_pipeline.Pipeline([("scaler", scaler),("model", sub2model_model)])
                                                    sub4model_pipes[str(set_feature_ddrs)]=sub4model_pipe
                                                sub3model_pipes[key]=sub4model_pipes
                                        sub2model_pipes[sub3model_type]=sub3model_pipes
                                    # case "quadratic regression":
                                    #     sub3model_type="quadratic regression random quadratic regression"
                                    #     sub2model_model=sklearn_linear_model.LinearRegression()
                                    #     sub3model_parameters=sub2model_parameters[sub3model_type]
                                    #     sub3model_pipes={}
                                    #     for key in tqdm.tqdm(sub3model_parameters.keys(), desc=f"Generating pipes for {sub3model_type} matrix of features DDRs: \n"):
                                    #         if key not in ["number of features", "number of informative features", "number of samples", "random state"]:
                                    #             sets_feature_ddrs=sub3model_parameters[key]
                                    #             sub4model_pipes={}
                                    #             for set_feature_ddrs in tqdm.tqdm(sets_feature_ddrs, desc=f"Generating pipes for {key} sets feature DDRs: \n"):
                                    #                 scaler= Set_Feature_DDRs_Invariant_Standard_Scaler(set_feature_ddrs)
                                    #                 sub4model_pipe=sklearn_pipeline.Pipeline([("scaler", scaler),("model", sub2model_model)])
                                    #                 sub4model_pipes[str(set_feature_ddrs)]=sub4model_pipe
                                    #             sub3model_pipes[key]=sub4model_pipes
                                    #     sub2model_pipes[sub3model_type]=sub3model_pipes
                                    case "linear support vector classifier":
                                        sub3model_type="linear support vector classifier random 2-class classification"
                                        C=sub2model_parameters["C"]
                                        random_state=sub2model_parameters["random state"]
                                        sub2model_model=sklearn_svm.LinearSVC(C=C, random_state=random_state)
                                        sub3model_parameters=sub2model_parameters[sub3model_type]
                                        sub3model_pipes={}
                                        for key in tqdm.tqdm(sub3model_parameters.keys(), desc=f"Generating pipes for {sub3model_type} matrix of features DDRs: \n"):
                                            if key not in ["number of features", "number of samples", "number of informative features", "number of redundant features", "number of classes", "number of clusters per class", "random state"]:
                                                sets_feature_ddrs=sub3model_parameters[key]
                                                sub4model_pipes={}
                                                for set_feature_ddrs in tqdm.tqdm(sets_feature_ddrs, desc=f"Generating pipes for {key} sets feature DDRs: \n"):
                                                    scaler= Set_Feature_DDRs_Invariant_Standard_Scaler(set_feature_ddrs)
                                                    sub4model_pipe=sklearn_pipeline.Pipeline([("scaler", scaler),("model", sub2model_model)])
                                                    sub4model_pipes[str(set_feature_ddrs)]=sub4model_pipe
                                                sub3model_pipes[key]=sub4model_pipes
                                        sub2model_pipes[sub3model_type]=sub3model_pipes
                                    case "linear support vector regressor":
                                        sub3model_type="linear support vector regressor random regression"
                                        epsilon=sub2model_parameters["epsilon"]
                                        C=sub2model_parameters["C"]
                                        random_state=sub2model_parameters["random state"]
                                        sub2model_model=sklearn_svm.LinearSVR(epsilon=epsilon, C=C, random_state=random_state)
                                        sub3model_parameters=sub2model_parameters[sub3model_type]
                                        sub3model_pipes={}
                                        for key in tqdm.tqdm(sub3model_parameters.keys(), desc=f"Generating pipes for {sub3model_type} matrix of features DDRs: \n"):
                                            if key not in ["number of features", "number of informative features", "number of samples", "random state"]:
                                                sets_feature_ddrs=sub3model_parameters[key]
                                                sub4model_pipes={}
                                                for set_feature_ddrs in tqdm.tqdm(sets_feature_ddrs, desc=f"Generating pipes for {key} sets feature DDRs: \n"):
                                                    scaler= Set_Feature_DDRs_Invariant_Standard_Scaler(set_feature_ddrs)
                                                    sub4model_pipe=sklearn_pipeline.Pipeline([("scaler", scaler),("model", sub2model_model)])
                                                    sub4model_pipes[str(set_feature_ddrs)]=sub4model_pipe
                                                sub3model_pipes[key]=sub4model_pipes
                                        sub2model_pipes[sub3model_type]=sub3model_pipes
                                    case "k nearest neighbors classifier":
                                        sub3model_type="k nearest neighbors classifier random 2-class classification"
                                        n_neighbors=sub2model_parameters["number of neighbors"]
                                        sub2model_model=sklearn_neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
                                        sub3model_parameters=sub2model_parameters[sub3model_type]
                                        sub3model_pipes={}
                                        for key in tqdm.tqdm(sub3model_parameters.keys(), desc=f"Generating pipes for {sub3model_type} matrix of features DDRs: \n"):
                                            if key not in ["number of features", "number of samples", "number of informative features", "number of redundant features", "number of classes", "number of clusters per class", "random state"]:
                                                sets_feature_ddrs=sub3model_parameters[key]
                                                sub4model_pipes={}
                                                for set_feature_ddrs in tqdm.tqdm(sets_feature_ddrs, desc=f"Generating pipes for {key} sets feature DDRs: \n"):
                                                    scaler= Set_Feature_DDRs_Invariant_Standard_Scaler(set_feature_ddrs)
                                                    sub4model_pipe=sklearn_pipeline.Pipeline([("scaler", scaler),("model", sub2model_model)])
                                                    sub4model_pipes[str(set_feature_ddrs)]=sub4model_pipe
                                                sub3model_pipes[key]=sub4model_pipes
                                        sub2model_pipes[sub3model_type]=sub3model_pipes
                                    case "k nearest neighbors regressor":
                                        sub3model_type="k nearest neighbors regressor friedman #1"
                                        n_neighbors=sub2model_parameters["number of neighbors"]
                                        sub2model_model=sklearn_neighbors.KNeighborsRegressor(n_neighbors=n_neighbors)
                                        sub3model_parameters=sub2model_parameters[sub3model_type]
                                        sub3model_pipes={}
                                        for key in tqdm.tqdm(sub3model_parameters.keys(), desc=f"Generating pipes for {sub3model_type} matrix of features DDRs: \n"):
                                            if key not in ["number of features","number of samples", "random state"]:
                                                sets_feature_ddrs=sub3model_parameters[key]
                                                sub4model_pipes={}
                                                for set_feature_ddrs in tqdm.tqdm(sets_feature_ddrs, desc=f"Generating pipes for {key} sets feature DDRs: \n"):
                                                    scaler= Set_Feature_DDRs_Invariant_Standard_Scaler(set_feature_ddrs)
                                                    sub4model_pipe=sklearn_pipeline.Pipeline([("scaler", scaler),("model", sub2model_model)])
                                                    sub4model_pipes[str(set_feature_ddrs)]=sub4model_pipe
                                                sub3model_pipes[key]=sub4model_pipes
                                        sub2model_pipes[sub3model_type]=sub3model_pipes
                                    # case "random forest classifier":
                                    #     sub3model_type="random forest classifier random 2-class classification"
                                    #     n_estimators=sub2model_parameters["number of estimators"]
                                    #     max_features=sub2model_parameters["maximum number of features"]
                                    #     max_depth=sub2model_parameters["maximum depth"]
                                    #     min_samples_split=sub2model_parameters["minimum number of samples required to split an internal node"]
                                    #     random_state=sub2model_parameters["random state"]
                                    #     sub2model_model=sklearn_ensemble.RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
                                    # case "random forest regressor":
                                    #     sub3model_type="random forest regressor friedman #1"
                                    #     n_estimators=sub2model_parameters["number of estimators"]
                                    #     max_features=sub2model_parameters["maximum number of features"]
                                    #     max_depth=sub2model_parameters["maximum depth"]
                                    #     min_samples_split=sub2model_parameters["minimum number of samples required to split an internal node"]
                                    #     random_state=sub2model_parameters["random state"]
                                    #     sub2model_model=sklearn_ensemble.RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
                                    #     sub3model_parameters=sub2model_parameters[sub3model_type]
                                    #     sub3model_pipes={}
                                    #     for key in tqdm.tqdm(sub3model_parameters.keys(), desc=f"Generating pipes for {sub3model_type} matrix of features DDRs: \n"):
                                    #         if key not in ["number of features", "number of samples", "random state"]:
                                    #             sets_feature_ddrs=sub3model_parameters[key]
                                    #             sub4model_pipes={}
                                    #             for set_feature_ddrs in tqdm.tqdm(sets_feature_ddrs, desc=f"Generating pipes for {key} sets feature DDRs: \n"):
                                    #                 scaler= Set_Feature_DDRs_Invariant_Standard_Scaler(set_feature_ddrs)
                                    #                 sub4model_pipe=sklearn_pipeline.Pipeline([("scaler", scaler),("model", sub2model_model)])
                                    #                 sub4model_pipes[str(set_feature_ddrs)]=sub4model_pipe
                                    #             sub3model_pipes[key]=sub4model_pipes
                                    #     sub2model_pipes[sub3model_type]=sub3model_pipes
                                    case "multi-layer perceptron classifier":
                                        sub3model_type="multi-layer perceptron classifier random 2-class classification"
                                        layer_widths=sub2model_parameters["layer widths"]
                                        sub2model_model=skorch.NeuralNetClassifier(module=MultiLayerPerceptron, module__layer_widths=layer_widths, module__activation_function=torch.nn.ReLU(), criterion=torch.nn.CrossEntropyLoss())
                                        sub3model_parameters=sub2model_parameters[sub3model_type]
                                        sub3model_pipes={}
                                        for key in tqdm.tqdm(sub3model_parameters.keys(), desc=f"Generating pipes for {sub3model_type} matrix of features DDRs: \n"):
                                            if key not in ["number of features", "number of samples", "number of informative features", "number of redundant features", "number of classes", "number of clusters per class", "random state"]:
                                                sets_feature_ddrs=sub3model_parameters[key]
                                                sub4model_pipes={}
                                                for set_feature_ddrs in tqdm.tqdm(sets_feature_ddrs, desc=f"Generating pipes for {key} sets feature DDRs: \n"):
                                                    scaler= Set_Feature_DDRs_Invariant_Standard_Scaler(set_feature_ddrs)
                                                    sub4model_pipe=sklearn_pipeline.Pipeline([("scaler", scaler),("model", sub2model_model)])
                                                    sub4model_pipes[str(set_feature_ddrs)]=sub4model_pipe
                                                sub3model_pipes[key]=sub4model_pipes
                                        sub2model_pipes[sub3model_type]=sub3model_pipes
                                    case "multi-layer perceptron regressor":
                                        sub3model_type="multi-layer perceptron regressor friedman #1"
                                        layer_widths=sub2model_parameters["layer widths"]
                                        sub2model_model=skorch.NeuralNetRegressor(module=MultiLayerPerceptron, module__layer_widths=layer_widths, module__activation_function=torch.nn.ReLU(), criterion=torch.nn.MSELoss())                            
                                        sub3model_parameters=sub2model_parameters[sub3model_type]
                                        sub3model_pipes={}
                                        for key in tqdm.tqdm(sub3model_parameters.keys(), desc=f"Generating pipes for {sub3model_type} matrix of features DDRs: \n"):
                                            if key not in ["number of features", "number of samples", "random state"]:
                                                sets_feature_ddrs=sub3model_parameters[key]
                                                sub4model_pipes={}
                                                for set_feature_ddrs in tqdm.tqdm(sets_feature_ddrs, desc=f"Generating pipes for {key} sets feature DDRs: \n"):
                                                    scaler= Set_Feature_DDRs_Invariant_Standard_Scaler(set_feature_ddrs)
                                                    sub4model_pipe=sklearn_pipeline.Pipeline([("scaler", scaler),("model", sub2model_model)])
                                                    sub4model_pipes[str(set_feature_ddrs)]=sub4model_pipe
                                                sub3model_pipes[key]=sub4model_pipes
                                        sub2model_pipes[sub3model_type]=sub3model_pipes
                                for sub3model_type in sub3model_types:
                                    sub3model_parameters=sub2model_parameters[sub3model_type]
                                    #sub3model_pipes=Sub3model_Pipes(sub3model_parameters, sub3model_type)
                                    #sub2model_pipes[sub3model_type]=sub3model_pipes
                                
                                
                                #sub2model_pipe=sklearn_pipeline.Pipeline([("scaler", scaler), ("model", sub2model_model)])
                                #sub2model_pipes[sub2model_type]=sub2model_pipe
                                return sub2model_pipes
                            
                            submodel_pipes={}
                            sub2model_types=[]
                            match submodel_type:
                                case "ordinary least squares":
                                    sub2model_type="ordinary least squares random regression"
                                    submodel_model=sklearn_linear_model.LinearRegression()
                                    sub2model_parameters=submodel_parameters[sub2model_type]
                                    sub2model_pipes={}
                                    for key in tqdm.tqdm(sub2model_parameters.keys(), desc=f"Generating pipes for {sub2model_type} matrix of features DDRs: \n"):
                                        if key not in ["number of features", "number of informative features", "number of samples", "random state"]:
                                            sets_feature_ddrs=sub2model_parameters[key]
                                            sub3model_pipes={}
                                            for set_feature_ddrs in tqdm.tqdm(sets_feature_ddrs, desc=f"Generating pipes for {key} sets feature DDRs: \n"):
                                                scaler= Set_Feature_DDRs_Invariant_Standard_Scaler(set_feature_ddrs)
                                                sub3model_pipe=sklearn_pipeline.Pipeline([("scaler", scaler),("model", submodel_model)])
                                                sub3model_pipes[str(set_feature_ddrs)]=sub3model_pipe
                                            sub2model_pipes[key]=sub3model_pipes
                                    submodel_pipes[sub2model_type]=sub2model_pipes
                                # case "ridge regression and classification":
                                #     sub2model_types=["ridge regressor", "ridge classifier"]
                                # case "lasso":
                                #     sub2model_types=["lasso random regression sparse uncorrelated"]
                                #     alpha=submodel_parameters["alpha"]
                                #     submodel_model=sklearn_linear_model.Lasso(alpha=alpha)
                                # case "elastic-net":
                                #     alpha=submodel_parameters["alpha"]
                                #     l1_ratio=submodel_parameters["l1 ratio"]
                                #     submodel_pipe=sklearn_linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
                                #     submodel_pipes[submodel_type]=submodel_pipe
                                case "logistic regression":
                                    sub2model_types=["binary logistic regression"]
                                # case "polynomial regression":
                                #     sub2model_types=["quadratic regression"]
                                case "support vector classifier":
                                    sub2model_types=["linear support vector classifier"]
                                case "support vector regressor":
                                    sub2model_types=["linear support vector regressor"]
                                case "nearest neighbors classifier":
                                    sub2model_types=["k nearest neighbors classifier"]
                                case "nearest neighbors regressor":
                                    sub2model_types=["k nearest neighbors regressor"]
                                case "decision tree classifier":
                                    max_depth=submodel_parameters["maximum depth"]
                                    min_samples_leaf=submodel_parameters["minimum number of samples required to be at a leaf"]
                                    random_state=submodel_parameters["random state"]
                                    sub2model_type="decision tree classifier random 2-class classification"
                                    submodel_model=sklearn_tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
                                    sub2model_parameters=submodel_parameters[sub2model_type]
                                    sub2model_pipes={}
                                    for key in tqdm.tqdm(sub2model_parameters.keys(), desc=f"Generating pipes for {sub2model_type} matrix of features DDRs: \n"):
                                        if key not in ["number of features", "number of samples", "number of informative features", "number of redundant features", "number of classes", "number of clusters per class", "random state"]:
                                            sets_feature_ddrs=sub2model_parameters[key]
                                            sub3model_pipes={}
                                            for set_feature_ddrs in tqdm.tqdm(sets_feature_ddrs, desc=f"Generating pipes for {key} sets of feature DDRs: \n"):
                                                scaler= Set_Feature_DDRs_Invariant_Standard_Scaler(set_feature_ddrs)
                                                sub3model_pipe=sklearn_pipeline.Pipeline([("scaler", scaler),("model", submodel_model)])
                                                sub3model_pipes[str(set_feature_ddrs)]=sub3model_pipe
                                            sub2model_pipes[key]=sub3model_pipes                                
                                    submodel_pipes[sub2model_type]=sub2model_pipes
                                case "decision tree regressor":
                                    max_depth=submodel_parameters["maximum depth"]
                                    min_samples_leaf=submodel_parameters["minimum number of samples required to be at a leaf"]
                                    random_state=submodel_parameters["random state"]
                                    sub2model_type="decision tree regressor friedman #1"
                                    submodel_model=sklearn_tree.DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
                                    sub2model_parameters=submodel_parameters[sub2model_type]
                                    sub2model_pipes={}
                                    for key in tqdm.tqdm(sub2model_parameters.keys(), desc=f"Generating pipes for {sub2model_type} matrix of features DDRs: \n"):
                                        if key not in ["number of features", "number of samples", "random state"]:
                                            sets_feature_ddrs=sub2model_parameters[key]
                                            sub3model_pipes={}
                                            for set_feature_ddrs in tqdm.tqdm(sets_feature_ddrs, desc=f"Generating pipes for {key} sets of feature DDRs: \n"):
                                                scaler= Set_Feature_DDRs_Invariant_Standard_Scaler(set_feature_ddrs)
                                                sub3model_pipe=sklearn_pipeline.Pipeline([("scaler", scaler),("model", submodel_model)])
                                                sub3model_pipes[str(set_feature_ddrs)]=sub3model_pipe
                                            sub2model_pipes[key]=sub3model_pipes
                                    submodel_pipes[sub2model_type]=sub2model_pipes
                                # case "random forests":
                                #     sub2model_types=["random forest classifier", "random forest regressor"]
                                case "multi-layer perceptrons":
                                    sub2model_types=["multi-layer perceptron classifier", "multi-layer perceptron regressor"]
                                case "k means":
                                    n_clusters=submodel_parameters["number of clusters"]
                                    random_state=submodel_parameters["random state"]
                                    sub2model_type="k means isotropic Gaussian blobs"
                                    submodel_model=sklearn_cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
                                    sub2model_parameters=submodel_parameters[sub2model_type]
                                    sub2model_pipes={}
                                    for key in tqdm.tqdm(sub2model_parameters.keys(), desc=f"Generating pipes for {sub2model_type} matrix of features DDRs: \n"):
                                        if key not in ["number of centers","number of features", "number of samples", "random state", "standard deviation of centers"]:
                                            sets_feature_ddrs=sub2model_parameters[key]
                                            sub3model_pipes={}
                                            for set_feature_ddrs in tqdm.tqdm(sets_feature_ddrs, desc=f"Generating pipes for {key} sets of feature DDRs: \n"):
                                                scaler= Set_Feature_DDRs_Invariant_Standard_Scaler(set_feature_ddrs)
                                                sub3model_pipe=sklearn_pipeline.Pipeline([("scaler", scaler),("model", submodel_model)])
                                                sub3model_pipes[str(set_feature_ddrs)]=sub3model_pipe
                                            sub2model_pipes[key]=sub3model_pipes
                                    submodel_pipes[sub2model_type]=sub2model_pipes
                            for sub2model_type in sub2model_types:
                                sub2model_parameters=submodel_parameters[sub2model_type]
                                sub2model_pipes=Sub2model_Pipes(sub2model_parameters, sub2model_type)
                                submodel_pipes[sub2model_type]=sub2model_pipes
                            return submodel_pipes
                    
                        model_pipes={}
                        submodel_types=[]
                        match model_type:
                            case "linear models":
                                submodel_types=["ordinary least squares", "ridge regression and classification", "lasso", "elastic-net", "logistic regression", "polynomial regression"]
                            # case "kernel ridge regression":
                            #     alpha=model_parameters["alpha"]
                            #     kernel=model_parameters["kernel"]
                            #     gamma=model_parameters["gamma"]
                            #     coef0=model_parameters["coef0"]
                            #     model_pipe=sklearn_kernel_ridge.KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma, coef0=coef0)
                            case "support vector machines":
                                submodel_types=["support vector classifier", "support vector regressor"]
                            case "supervised nearest neighbors":
                                submodel_types=["nearest neighbors classifier", "nearest neighbors regressor"]
                            case "decision trees":
                                submodel_types=["decision tree classifier", "decision tree regressor"]
                            # case "ensembles":
                            #     submodel_types=["random forests"]
                            case "neural network models":
                                submodel_types=["multi-layer perceptrons"]
                            case "clustering":
                                submodel_types=["k means"]
                        for submodel_type in submodel_types:
                            submodel_parameters=model_parameters[submodel_type]
                            submodel_pipes=Submodel_Pipes(submodel_parameters, submodel_type)
                            model_pipes[submodel_type]=submodel_pipes
                        return model_pipes
                    
                    learning_pipes={}
                    model_types=[]
                    if learning_type=="supervised learning":
                        model_types=["linear models", "kernel ridge regression", "support vector machines", "supervised nearest neighbors", "decision trees", "ensembles", "neural network models"]
                    elif learning_type=="unsupervised learning":
                        model_types=["clustering"]
                    for model_type in model_types:
                        model_parameters=learning_parameters[model_type]
                        model_pipes=Model_Pipes(model_parameters, model_type)
                        learning_pipes[model_type]=model_pipes             
                    return learning_pipes
    
                pipes={}
                learning_types=["supervised learning", "unsupervised learning"]
                for learning_type in learning_types:
                    learning_parameters=parameters[learning_type]
                    learning_pipes=Learning_Pipes(learning_parameters, learning_type)
                    pipes[learning_type]=learning_pipes
                return pipes
            
            inputs={}
            datasets=Datasets(parameters)
            pipes=Pipes(parameters)
            inputs["parameters"]=parameters
            inputs["datasets"]=datasets
            inputs["pipes"]=pipes
            return inputs
        
        def Outputs(inputs):
            """
            Takes the inputs for the experiment and returns the outputs.
    
            Parameters
            ----------
            inputs : Dict
                The inputs for the experiment.
    
            Returns
            -------
            outputs : Dict
                The outputs for the experiment.
    
            """
            
            def Scores(datasets, pipes):
                """
                Takes the datasets and pipes as inputs and returns the scores.
    
                Parameters
                ----------
                datasets : Dict
                    Datasets for the experiment.
                pipes : Dict
                    Pipelines for the experiment.
    
                Returns
                -------
                scores : Dict
                    Scores from the experiment.
    
                """
                def Learning_Scores(learning_datasets, learning_pipes):
                    """
                    Takes the learning datasets and learning pipes as inputs and returns the learning scores.
    
                    Parameters
                    ----------
                    learning_datasets : Dict
                        Datasets for the learning type.
                    learning_pipes : Dict
                        Pipelines for the learning_type.
    
                    Returns
                    -------
                    learning_scores : Dict
                        Scores from the learning type.
    
                    """
                    
                    def Model_Scores(model_datasets, model_pipes):
                        """
                        Takes the model datasets and model pipes and return the model scores
    
                        Parameters
                        ----------
                        model_datasets : Dict
                            Datasets for the model.
                        model_pipes : Dict
                            Pipelines for the model.
    
                        Returns
                        -------
                        model_scores : Dict
                            Scores from the model.
    
                        """
                        
                        def Submodel_Scores(submodel_datasets, submodel_pipes):
                            """
                            Takes the submodel datasets and submodel pipes and returns the submodel scores
    
                            Parameters
                            ----------
                            submodel_datasets : Dict
                                Datasets for the submodel.
                            submodel_pipes : Dict
                                Pipelines for the submodel.
    
                            Returns
                            -------
                            submodel_scores : Dict
                                Scores from the submodel.
    
                            """
                        
                            def Sub2model_Scores(sub2model_datasets, sub2model_pipes):
                                """
                                Takes the sub2model datasets and sub2model pipes and returns the sub2model scores
    
                                Parameters
                                ----------
                                sub2model_datasets : Dict
                                    Datasets for the sub2model.
                                sub2model_pipes : Dict
                                    Pipelines for the sub2model.
    
                                Returns
                                -------
                                sub2model_scores : Dict
                                    Scores from the sub2model.
    
                                """
                                
                                def Sub3model_Scores(sub3model_datasets, sub3model_pipes):
                                    """
                                    Takes the sub3model datasets and sub3model pipes and returns the sub3model scores
    
                                    Parameters
                                    ----------
                                    sub3model_datasets : Dict
                                        Datasets for the sub3model.
                                    sub3model_pipes : Dict
                                        Pipelines for the sub3model.
    
                                    Returns
                                    -------
                                    sub3model_scores : Dict
                                        Scores from the sub3model.
    
                                    """
                                    
                                    sub3model_scores={}
                                    match sub3model_type:
                                        # case "quadratic regression random quadratic regression":
                                        #     0==0
                                        #     # feature_matrix_test=sub3model_datasets["testing matrix of features"]
                                        #     # targets_test=sub3model_datasets["testing targets"]
                                        #     # targets_test_range=max(targets_test)-min(targets_test)
                                        #     # feature_matrix_train=sub3model_datasets["training matrix of features"]
                                        #     # targets_train=sub3model_datasets["training targets"]
                                        #     # targets_train_range=max(targets_train)-min(targets_train)
                                        #     # sub4model_pipe_types=sub3model_pipes.keys()
                                        #     # for i,sub4model_pipe_type in tqdm.tqdm(enumerate(sub4model_pipe_types),desc=f"Getting scores for {sub3model_type}"):
                                        #     #     sub4model_scores={}
                                        #     #     sub4model_pipes=sub3model_pipes[sub4model_pipe_type]
                                        #     #     sub5model_pipe_types=sub4model_pipes.keys()
                                        #     #     for sub5model_pipe_type in sub5model_pipe_types:
                                        #     #         sub5model_pipe=sub4model_pipes[sub5model_pipe_type]
                                        #     #         scaler_pipe=sub5model_pipe[:1]
                                        #     #         if i%int((N_FEATURE_MATRIX_DDRS/1))==0:
                                        #     #             fig, ax=plt.subplots()
                                        #     #             ax.set_title(f"{sub3model_type} with matrix of features DDR == {sub5model_pipe_type}")
                                        #     #             ax.set_xlabel("matrix of features DDR-invariant scaled feature")
                                        #     #             ax.set_ylabel("target")
                                        #     #             ax.grid=True
                                        #     #             feature_matrix_train_plot=feature_matrix_train
                                        #     #             feature_matrix_train_plot_scaled=scaler_pipe.fit_transform(feature_matrix_train_plot)
                                        #     #             feature_matrix_test_plot=feature_matrix_test
                                        #     #             feature_matrix_test_plot_scaled=scaler_pipe.fit_transform(feature_matrix_test_plot)
                                        #     #             ax.scatter(feature_matrix_train_plot_scaled, targets_train, label="train")
                                        #     #             ax.scatter(feature_matrix_test_plot_scaled, targets_test, label="test")
                                        #     #             ax.legend()
                                        #     #         sub5model_pipe=sub5model_pipe.fit(feature_matrix_train, targets_train)
                                        #     #         targets_train_pred=sub5model_pipe.predict(feature_matrix_train)
                                        #     #         targets_test_pred=sub5model_pipe.predict(feature_matrix_test)
                                        #     #         targets_train_mean_squared_error=sklearn_metrics.mean_squared_error(targets_train, targets_train_pred)
                                        #     #         targets_train_normalized_mean_squared_error=targets_train_mean_squared_error/targets_train_range**2
                                        #     #         sub5model_score_train=1-targets_train_normalized_mean_squared_error
                                        #     #         sub4model_scores[sub5model_pipe_type+" train"]=sub5model_score_train        
                                        #     #         targets_test_mean_squared_error=sklearn_metrics.mean_squared_error(targets_test, targets_test_pred)
                                        #     #         targets_test_normalized_mean_squared_error=targets_test_mean_squared_error/targets_test_range**2
                                        #     #         sub5model_score_test=1-targets_test_normalized_mean_squared_error
                                        #     #         sub4model_scores[sub5model_pipe_type+" test"]=sub5model_score_test
                                        #     #         plt.show()
                                        #     #         plt.close()
                                        #     #     sub3model_scores[sub4model_pipe_type]=sub4model_scores
                                        case "binary logistic regression random 2-class classification":
                                            feature_matrix_test=sub3model_datasets["testing matrix of features"]
                                            targets_test=sub3model_datasets["testing targets"]
                                            targets_test_range=max(targets_test)-min(targets_test)
                                            feature_matrix_train=sub3model_datasets["training matrix of features"]
                                            targets_train=sub3model_datasets["training targets"]
                                            targets_train_range=max(targets_train)-min(targets_train)
                                            sub4model_pipe_types=sub3model_pipes.keys()
                                            for i,sub4model_pipe_type in tqdm.tqdm(enumerate(sub4model_pipe_types),desc=f"Getting scores for {sub3model_type}"):
                                                sub4model_scores={}
                                                sub4model_pipes=sub3model_pipes[sub4model_pipe_type]
                                                sub5model_pipe_types=sub4model_pipes.keys()
                                                for sub5model_pipe_type in sub5model_pipe_types:
                                                    sub5model_pipe=sub4model_pipes[sub5model_pipe_type]
                                                    sub5model_pipe=sub5model_pipe.fit(feature_matrix_train, targets_train)
                                                    targets_train_pred=sub5model_pipe.predict(feature_matrix_train)
                                                    targets_test_pred=sub5model_pipe.predict(feature_matrix_test)
                                                    sub5model_score_train=sklearn_metrics.f1_score(targets_train, targets_train_pred)
                                                    sub4model_scores[sub5model_pipe_type+" train"]=sub5model_score_train        
                                                    sub5model_score_test=sklearn_metrics.f1_score(targets_test, targets_test_pred)
                                                    sub4model_scores[sub5model_pipe_type+" test"]=sub5model_score_test
                                                sub3model_scores[sub4model_pipe_type]=sub4model_scores
                                        case "linear support vector regressor random regression":
                                            feature_matrix_test=sub3model_datasets["testing matrix of features"]
                                            targets_test=sub3model_datasets["testing targets"]
                                            targets_test_range=max(targets_test)-min(targets_test)
                                            feature_matrix_train=sub3model_datasets["training matrix of features"]
                                            targets_train=sub3model_datasets["training targets"]
                                            targets_train_range=max(targets_train)-min(targets_train)
                                            sub4model_pipe_types=sub3model_pipes.keys()
                                            for i,sub4model_pipe_type in tqdm.tqdm(enumerate(sub4model_pipe_types),desc=f"Getting scores for {sub3model_type}"):
                                                sub4model_scores={}
                                                sub4model_pipes=sub3model_pipes[sub4model_pipe_type]
                                                sub5model_pipe_types=sub4model_pipes.keys()
                                                for sub5model_pipe_type in sub5model_pipe_types:
                                                    sub5model_pipe=sub4model_pipes[sub5model_pipe_type]
                                                    sub5model_pipe=sub5model_pipe.fit(feature_matrix_train, targets_train)
                                                    targets_train_pred=sub5model_pipe.predict(feature_matrix_train)
                                                    targets_test_pred=sub5model_pipe.predict(feature_matrix_test)
                                                    targets_train_mean_squared_error=sklearn_metrics.mean_squared_error(targets_train, targets_train_pred)
                                                    targets_train_normalized_mean_squared_error=targets_train_mean_squared_error/targets_train_range**2
                                                    sub5model_score_train=1-targets_train_normalized_mean_squared_error
                                                    sub4model_scores[sub5model_pipe_type+" train"]=sub5model_score_train        
                                                    targets_test_mean_squared_error=sklearn_metrics.mean_squared_error(targets_test, targets_test_pred)
                                                    targets_test_normalized_mean_squared_error=targets_test_mean_squared_error/targets_test_range**2
                                                    sub5model_score_test=1-targets_test_normalized_mean_squared_error
                                                    sub4model_scores[sub5model_pipe_type+" test"]=sub5model_score_test
                                                sub3model_scores[sub4model_pipe_type]=sub4model_scores
                                        case "linear support vector classifier random 2-class classification":
                                            feature_matrix_test=sub3model_datasets["testing matrix of features"]
                                            targets_test=sub3model_datasets["testing targets"]
                                            targets_test_range=max(targets_test)-min(targets_test)
                                            feature_matrix_train=sub3model_datasets["training matrix of features"]
                                            targets_train=sub3model_datasets["training targets"]
                                            targets_train_range=max(targets_train)-min(targets_train)
                                            sub4model_pipe_types=sub3model_pipes.keys()
                                            for i,sub4model_pipe_type in tqdm.tqdm(enumerate(sub4model_pipe_types),desc=f"Getting scores for {sub3model_type}"):
                                                sub4model_scores={}
                                                sub4model_pipes=sub3model_pipes[sub4model_pipe_type]
                                                sub5model_pipe_types=sub4model_pipes.keys()
                                                for sub5model_pipe_type in sub5model_pipe_types:
                                                    sub5model_pipe=sub4model_pipes[sub5model_pipe_type]
                                                    sub5model_pipe=sub5model_pipe.fit(feature_matrix_train, targets_train)
                                                    targets_train_pred=sub5model_pipe.predict(feature_matrix_train)
                                                    targets_test_pred=sub5model_pipe.predict(feature_matrix_test)
                                                    sub5model_score_train=sklearn_metrics.f1_score(targets_train, targets_train_pred)
                                                    sub4model_scores[sub5model_pipe_type+" train"]=sub5model_score_train        
                                                    sub5model_score_test=sklearn_metrics.f1_score(targets_test, targets_test_pred)
                                                    sub4model_scores[sub5model_pipe_type+" test"]=sub5model_score_test
                                                sub3model_scores[sub4model_pipe_type]=sub4model_scores
                                        case "k nearest neighbors regressor friedman #1":
                                            feature_matrix_test=sub3model_datasets["testing matrix of features"]
                                            targets_test=sub3model_datasets["testing targets"]
                                            targets_test_range=max(targets_test)-min(targets_test)
                                            feature_matrix_train=sub3model_datasets["training matrix of features"]
                                            targets_train=sub3model_datasets["training targets"]
                                            targets_train_range=max(targets_train)-min(targets_train)
                                            sub4model_pipe_types=sub3model_pipes.keys()
                                            for i,sub4model_pipe_type in tqdm.tqdm(enumerate(sub4model_pipe_types),desc=f"Getting scores for {sub3model_type}"):
                                                sub4model_scores={}
                                                sub4model_pipes=sub3model_pipes[sub4model_pipe_type]
                                                sub5model_pipe_types=sub4model_pipes.keys()
                                                for sub5model_pipe_type in sub5model_pipe_types:
                                                    sub5model_pipe=sub4model_pipes[sub5model_pipe_type]
                                                    sub5model_pipe=sub5model_pipe.fit(feature_matrix_train, targets_train)
                                                    targets_train_pred=sub5model_pipe.predict(feature_matrix_train)
                                                    targets_test_pred=sub5model_pipe.predict(feature_matrix_test)
                                                    targets_train_mean_squared_error=sklearn_metrics.mean_squared_error(targets_train, targets_train_pred)
                                                    targets_train_normalized_mean_squared_error=targets_train_mean_squared_error/targets_train_range**2
                                                    sub5model_score_train=1-targets_train_normalized_mean_squared_error
                                                    sub4model_scores[sub5model_pipe_type+" train"]=sub5model_score_train        
                                                    targets_test_mean_squared_error=sklearn_metrics.mean_squared_error(targets_test, targets_test_pred)
                                                    targets_test_normalized_mean_squared_error=targets_test_mean_squared_error/targets_test_range**2
                                                    sub5model_score_test=1-targets_test_normalized_mean_squared_error
                                                    sub4model_scores[sub5model_pipe_type+" test"]=sub5model_score_test
                                                sub3model_scores[sub4model_pipe_type]=sub4model_scores
                                        case "k nearest neighbors classifier random 2-class classification":
                                            feature_matrix_test=sub3model_datasets["testing matrix of features"]
                                            targets_test=sub3model_datasets["testing targets"]
                                            targets_test_range=max(targets_test)-min(targets_test)
                                            feature_matrix_train=sub3model_datasets["training matrix of features"]
                                            targets_train=sub3model_datasets["training targets"]
                                            targets_train_range=max(targets_train)-min(targets_train)
                                            sub4model_pipe_types=sub3model_pipes.keys()
                                            for i,sub4model_pipe_type in tqdm.tqdm(enumerate(sub4model_pipe_types),desc=f"Getting scores for {sub3model_type}"):
                                                sub4model_scores={}
                                                sub4model_pipes=sub3model_pipes[sub4model_pipe_type]
                                                sub5model_pipe_types=sub4model_pipes.keys()
                                                for sub5model_pipe_type in sub5model_pipe_types:
                                                    sub5model_pipe=sub4model_pipes[sub5model_pipe_type]
                                                    sub5model_pipe=sub5model_pipe.fit(feature_matrix_train, targets_train)
                                                    targets_train_pred=sub5model_pipe.predict(feature_matrix_train)
                                                    targets_test_pred=sub5model_pipe.predict(feature_matrix_test)
                                                    sub5model_score_train=sklearn_metrics.f1_score(targets_train, targets_train_pred)
                                                    sub4model_scores[sub5model_pipe_type+" train"]=sub5model_score_train        
                                                    sub5model_score_test=sklearn_metrics.f1_score(targets_test, targets_test_pred)
                                                    sub4model_scores[sub5model_pipe_type+" test"]=sub5model_score_test
                                                sub3model_scores[sub4model_pipe_type]=sub4model_scores
                                        # case "random forest regressor friedman #1":
                                        #     feature_matrix_test=sub3model_datasets["testing matrix of features"]
                                        #     targets_test=sub3model_datasets["testing targets"]
                                        #     targets_test_range=max(targets_test)-min(targets_test)
                                        #     feature_matrix_train=sub3model_datasets["training matrix of features"]
                                        #     targets_train=sub3model_datasets["training targets"]
                                        #     targets_train_range=max(targets_train)-min(targets_train)
                                        #     sub4model_pipe_types=sub3model_pipes.keys()
                                        #     for i,sub4model_pipe_type in tqdm.tqdm(enumerate(sub4model_pipe_types),desc=f"Getting scores for {sub3model_type}"):
                                        #         sub4model_scores={}
                                        #         sub4model_pipes=sub3model_pipes[sub4model_pipe_type]
                                        #         sub5model_pipe_types=sub4model_pipes.keys()
                                        #         for sub5model_pipe_type in sub5model_pipe_types:
                                        #             sub5model_pipe=sub4model_pipes[sub5model_pipe_type]
                                        #             sub5model_pipe=sub5model_pipe.fit(feature_matrix_train, targets_train)
                                        #             targets_train_pred=sub5model_pipe.predict(feature_matrix_train)
                                        #             targets_test_pred=sub5model_pipe.predict(feature_matrix_test)
                                        #             targets_train_mean_squared_error=sklearn_metrics.mean_squared_error(targets_train, targets_train_pred)
                                        #             targets_train_normalized_mean_squared_error=targets_train_mean_squared_error/targets_train_range**2
                                        #             sub5model_score_train=1-targets_train_normalized_mean_squared_error
                                        #             sub4model_scores[sub5model_pipe_type+" train"]=sub5model_score_train        
                                        #             targets_test_mean_squared_error=sklearn_metrics.mean_squared_error(targets_test, targets_test_pred)
                                        #             targets_test_normalized_mean_squared_error=targets_test_mean_squared_error/targets_test_range**2
                                        #             sub5model_score_test=1-targets_test_normalized_mean_squared_error
                                        #             sub4model_scores[sub5model_pipe_type+" test"]=sub5model_score_test
                                        #         sub3model_scores[sub4model_pipe_type]=sub4model_scores
                                        case "multi-layer perceptron regressor friedman #1":
                                            feature_matrix_test=sub3model_datasets["testing matrix of features"]
                                            feature_matrix_test=torch.from_numpy(feature_matrix_test)
                                            targets_test=sub3model_datasets["testing targets"]
                                            targets_test=torch.from_numpy(targets_test)
                                            targets_test_range=torch.max(targets_test)-torch.min(targets_test)
                                            feature_matrix_train=sub3model_datasets["training matrix of features"]
                                            feature_matrix_train=torch.from_numpy(feature_matrix_train)
                                            targets_train=sub3model_datasets["training targets"]
                                            targets_train=torch.from_numpy(targets_train)
                                            targets_train_range=torch.max(targets_train)-torch.min(targets_train)
                                            sub4model_pipe_types=sub3model_pipes.keys()
                                            for i,sub4model_pipe_type in tqdm.tqdm(enumerate(sub4model_pipe_types),desc=f"Getting scores for {sub3model_type}"):
                                                sub4model_scores={}
                                                sub4model_pipes=sub3model_pipes[sub4model_pipe_type]
                                                sub5model_pipe_types=sub4model_pipes.keys()
                                                for sub5model_pipe_type in sub5model_pipe_types:
                                                    sub5model_pipe=sub4model_pipes[sub5model_pipe_type]
                                                    sub5model_pipe=sub5model_pipe.fit(feature_matrix_train, targets_train)
                                                    targets_train_pred=sub5model_pipe.predict(feature_matrix_train)
                                                    targets_train_pred=torch.from_numpy(targets_train_pred)
                                                    targets_test_pred=sub5model_pipe.predict(feature_matrix_test)
                                                    targets_test_pred=torch.from_numpy(targets_test_pred)
                                                    targets_train_mean_squared_error=torch.nn.MSELoss()(targets_train, targets_train_pred)
                                                    targets_train_normalized_mean_squared_error=targets_train_mean_squared_error/targets_train_range**2
                                                    sub5model_score_train=1-targets_train_normalized_mean_squared_error
                                                    sub4model_scores[sub5model_pipe_type+" train"]=sub5model_score_train        
                                                    targets_test_mean_squared_error=torch.nn.MSELoss()(targets_test, targets_test_pred)
                                                    targets_test_normalized_mean_squared_error=targets_test_mean_squared_error/targets_test_range**2
                                                    sub5model_score_test=1-targets_test_normalized_mean_squared_error
                                                    sub4model_scores[sub5model_pipe_type+" test"]=sub5model_score_test
                                                sub3model_scores[sub4model_pipe_type]=sub4model_scores
                                        case "multi-layer perceptron classifier random 2-class classification":
                                            feature_matrix_test=sub3model_datasets["testing matrix of features"]
                                            targets_test=sub3model_datasets["testing targets"]
                                            targets_test_range=max(targets_test)-min(targets_test)
                                            feature_matrix_train=sub3model_datasets["training matrix of features"]
                                            targets_train=sub3model_datasets["training targets"]
                                            targets_train_range=max(targets_train)-min(targets_train)
                                            sub4model_pipe_types=sub3model_pipes.keys()
                                            for i,sub4model_pipe_type in tqdm.tqdm(enumerate(sub4model_pipe_types),desc=f"Getting scores for {sub3model_type}"):
                                                sub4model_scores={}
                                                sub4model_pipes=sub3model_pipes[sub4model_pipe_type]
                                                sub5model_pipe_types=sub4model_pipes.keys()
                                                for sub5model_pipe_type in sub5model_pipe_types:
                                                    sub5model_pipe=sub4model_pipes[sub5model_pipe_type]
                                                    sub5model_pipe=sub5model_pipe.fit(feature_matrix_train.reshape(-1,1), torch.from_numpy(targets_train).long())
                                                    targets_train_pred=sub5model_pipe.predict(feature_matrix_train)
                                                    targets_test_pred=sub5model_pipe.predict(feature_matrix_test)
                                                    sub5model_score_train=torcheval_metrics.BinaryF1Score().update(torch.from_numpy(targets_train), torch.from_numpy(targets_train_pred)).compute()
                                                    sub4model_scores[sub5model_pipe_type+" train"]=sub5model_score_train        
                                                    sub5model_score_test=torcheval_metrics.BinaryF1Score().update(torch.from_numpy(targets_test), torch.from_numpy(targets_test_pred)).compute()
                                                    sub4model_scores[sub5model_pipe_type+" test"]=sub5model_score_test
                                                sub3model_scores[sub4model_pipe_type]=sub4model_scores
                                    return sub3model_scores 
                                        
                                sub2model_scores={}
                                sub3model_types=[]
                                match sub2model_type:
                                    case "ordinary least squares random regression":
                                        feature_matrix_test=sub2model_datasets["testing matrix of features"]
                                        targets_test=sub2model_datasets["testing targets"]
                                        targets_test_range=max(targets_test)-min(targets_test)
                                        feature_matrix_train=sub2model_datasets["training matrix of features"]
                                        targets_train=sub2model_datasets["training targets"]
                                        targets_train_range=max(targets_train)-min(targets_train)
                                        sub3model_pipe_types=sub2model_pipes.keys()
                                        for i,sub3model_pipe_type in tqdm.tqdm(enumerate(sub3model_pipe_types),desc=f"Getting scores for {sub2model_type}"):
                                            sub3model_scores={}
                                            sub3model_pipes=sub2model_pipes[sub3model_pipe_type]
                                            sub4model_pipe_types=sub3model_pipes.keys()
                                            for sub4model_pipe_type in sub4model_pipe_types:
                                                sub4model_pipe=sub3model_pipes[sub4model_pipe_type]
                                                sub4model_pipe=sub4model_pipe.fit(feature_matrix_train, targets_train)
                                                targets_train_pred=sub4model_pipe.predict(feature_matrix_train)
                                                targets_test_pred=sub4model_pipe.predict(feature_matrix_test)
                                                targets_train_mean_squared_error=sklearn_metrics.mean_squared_error(targets_train, targets_train_pred)
                                                targets_train_normalized_mean_squared_error=targets_train_mean_squared_error/targets_train_range**2
                                                sub4model_score_train=1-targets_train_normalized_mean_squared_error
                                                sub3model_scores[sub4model_pipe_type+" train"]=sub4model_score_train        
                                                targets_test_mean_squared_error=sklearn_metrics.mean_squared_error(targets_test, targets_test_pred)
                                                targets_test_normalized_mean_squared_error=targets_test_mean_squared_error/targets_test_range**2
                                                sub4model_score_test=1-targets_test_normalized_mean_squared_error
                                                sub3model_scores[sub4model_pipe_type+" test"]=sub4model_score_test
                                            sub2model_scores[sub3model_pipe_type]=sub3model_scores
                                    case "binary logistic regression":
                                        sub3model_types=["binary logistic regression random 2-class classification"]
                                    case "decision tree regressor friedman #1":
                                        feature_matrix_test=sub2model_datasets["testing matrix of features"]
                                        targets_test=sub2model_datasets["testing targets"]
                                        targets_test_range=max(targets_test)-min(targets_test)
                                        feature_matrix_train=sub2model_datasets["training matrix of features"]
                                        targets_train=sub2model_datasets["training targets"]
                                        targets_train_range=max(targets_train)-min(targets_train)
                                        sub3model_pipe_types=sub2model_pipes.keys()
                                        for i,sub3model_pipe_type in tqdm.tqdm(enumerate(sub3model_pipe_types),desc=f"Getting scores for {sub2model_type}"):
                                            sub3model_scores={}
                                            sub3model_pipes=sub2model_pipes[sub3model_pipe_type]
                                            sub4model_pipe_types=sub3model_pipes.keys()
                                            for sub4model_pipe_type in sub4model_pipe_types:
                                                sub4model_pipe=sub3model_pipes[sub4model_pipe_type]
                                                sub4model_pipe=sub4model_pipe.fit(feature_matrix_train, targets_train)
                                                targets_train_pred=sub4model_pipe.predict(feature_matrix_train)
                                                targets_test_pred=sub4model_pipe.predict(feature_matrix_test)
                                                targets_train_mean_squared_error=sklearn_metrics.mean_squared_error(targets_train, targets_train_pred)
                                                targets_train_normalized_mean_squared_error=targets_train_mean_squared_error/targets_train_range**2
                                                sub4model_score_train=1-targets_train_normalized_mean_squared_error
                                                sub3model_scores[sub4model_pipe_type+" train"]=sub4model_score_train        
                                                targets_test_mean_squared_error=sklearn_metrics.mean_squared_error(targets_test, targets_test_pred)
                                                targets_test_normalized_mean_squared_error=targets_test_mean_squared_error/targets_test_range**2
                                                sub4model_score_test=1-targets_test_normalized_mean_squared_error
                                                sub3model_scores[sub4model_pipe_type+" test"]=sub4model_score_test
                                            sub2model_scores[sub3model_pipe_type]=sub3model_scores
                                    case "decision tree classifier random 2-class classification":
                                        feature_matrix_test=sub2model_datasets["testing matrix of features"]
                                        targets_test=sub2model_datasets["testing targets"]
                                        feature_matrix_train=sub2model_datasets["training matrix of features"]
                                        targets_train=sub2model_datasets["training targets"]
                                        sub3model_pipe_types=sub2model_pipes.keys()
                                        for i,sub3model_pipe_type in tqdm.tqdm(enumerate(sub3model_pipe_types),desc=f"Getting scores for {sub2model_type}"):
                                            sub3model_scores={}
                                            sub3model_pipes=sub2model_pipes[sub3model_pipe_type]
                                            sub4model_pipe_types=sub3model_pipes.keys()
                                            for sub4model_pipe_type in sub4model_pipe_types:
                                                sub4model_pipe=sub3model_pipes[sub4model_pipe_type]
                                                sub4model_pipe=sub4model_pipe.fit(feature_matrix_train, targets_train)
                                                targets_train_pred=sub4model_pipe.predict(feature_matrix_train)
                                                targets_test_pred=sub4model_pipe.predict(feature_matrix_test)
                                                targets_train_f1_score=sklearn_metrics.f1_score(targets_train, targets_train_pred)
                                                sub3model_scores[sub4model_pipe_type+" train"]=targets_train_f1_score       
                                                targets_test_f1_score=sklearn_metrics.f1_score(targets_test, targets_test_pred)
                                                sub3model_scores[sub4model_pipe_type+" test"]=targets_test_f1_score
                                            sub2model_scores[sub3model_pipe_type]=sub3model_scores
                                    case "linear support vector regressor":
                                        sub3model_types=["linear support vector regressor random regression"]
                                    case "linear support vector classifier":
                                        sub3model_types=["linear support vector classifier random 2-class classification"]
                                    case "k nearest neighbors regressor":
                                        sub3model_types=["k nearest neighbors regressor friedman #1"]
                                    case "k nearest neighbors classifier":
                                        sub3model_types=["k nearest neighbors classifier random 2-class classification"]
                                    # case "random forest regressor":
                                    #     sub3model_types=["random forest regressor friedman #1"]
                                    case "multi-layer perceptron regressor":
                                        sub3model_types=["multi-layer perceptron regressor friedman #1"]
                                    case "multi-layer perceptron classifier":
                                        sub3model_types=["multi-layer perceptron classifier random 2-class classification"]
                                    case "k means isotropic Gaussian blobs":
                                        feature_matrix=sub2model_datasets["matrix of features"]
                                        targets=sub2model_datasets["targets"]
                                        targets_range=max(targets)-min(targets)
                                        sub3model_pipe_types=sub2model_pipes.keys()
                                        for i,sub3model_pipe_type in tqdm.tqdm(enumerate(sub3model_pipe_types),desc=f"Getting scores for {sub2model_type}"):
                                            sub3model_scores={}
                                            sub3model_pipes=sub2model_pipes[sub3model_pipe_type]
                                            sub4model_pipe_types=sub3model_pipes.keys()
                                            for sub4model_pipe_type in sub4model_pipe_types:
                                                sub4model_pipe=sub3model_pipes[sub4model_pipe_type]
                                                sub4model_pipe=sub4model_pipe.fit(feature_matrix, targets)
                                                targets_pred=sub4model_pipe.predict(feature_matrix)
                                                targets_v_measure_score=sklearn_metrics.v_measure_score(targets, targets_pred)
                                                sub3model_scores[sub4model_pipe_type]=targets_v_measure_score
                                            sub2model_scores[sub3model_pipe_type]=sub3model_scores
                                    # case "quadratic regression":
                                    #     sub3model_types=["quadratic regression random quadratic regression"]
                                for sub3model_type in sub3model_types:
                                    sub3model_datasets=sub2model_datasets[sub3model_type]
                                    sub3model_pipes=sub2model_pipes[sub3model_type]
                                    sub3model_scores=Sub3model_Scores(sub3model_datasets, sub3model_pipes)
                                    sub2model_scores[sub3model_type]=sub3model_scores
                                return sub2model_scores        
                        
                            submodel_scores={}
                            sub2model_types=[]
                            match submodel_type:
                                case "ordinary least squares":
                                    sub2model_types=["ordinary least squares random regression"]
                                case "logistic regression":
                                    sub2model_types=["binary logistic regression"]
                                case "decision tree regressor":
                                    sub2model_types=["decision tree regressor friedman #1"]
                                case "decision tree classifier":
                                    sub2model_types=["decision tree classifier random 2-class classification"]
                                case "support vector regressor":
                                    sub2model_types=["linear support vector regressor"]
                                case "support vector classifier":
                                    sub2model_types=["linear support vector classifier"]
                                case "nearest neighbors regressor":
                                    sub2model_types=["k nearest neighbors regressor"]
                                case "nearest neighbors classifier":
                                    sub2model_types=["k nearest neighbors classifier"]
                                # case "random forests":
                                #     sub2model_types=["random forest regressor"]
                                case "multi-layer perceptrons":
                                    sub2model_types=["multi-layer perceptron regressor", "multi-layer perceptron classifier"]
                                case "k means":
                                    sub2model_types=["k means isotropic Gaussian blobs"]
                                # case "polynomial regression":
                                #     sub2model_types=["quadratic regression"]
                            for sub2model_type in sub2model_types:
                                sub2model_datasets=submodel_datasets[sub2model_type]
                                sub2model_pipes=submodel_pipes[sub2model_type]
                                sub2model_scores=Sub2model_Scores(sub2model_datasets, sub2model_pipes)
                                submodel_scores[sub2model_type]=sub2model_scores
                            return submodel_scores
                        
                        model_scores={}
                        submodel_types=[]
                        match model_type:
                            case "linear models":
                                submodel_types=["ordinary least squares", "logistic regression", "polynomial regression"]
                            case "decision trees":
                                submodel_types=["decision tree regressor", "decision tree classifier"]
                            case "support vector machines":
                                submodel_types=["support vector regressor", "support vector classifier"]
                            case "supervised nearest neighbors":
                                submodel_types=["nearest neighbors regressor", "nearest neighbors classifier"]
                            # case "ensembles":
                            #     submodel_types=["random forests"]
                            case "neural network models":
                                submodel_types=["multi-layer perceptrons"]
                            case "clustering":
                                submodel_types=["k means"]
                        for submodel_type in submodel_types:
                            submodel_datasets=model_datasets[submodel_type]
                            submodel_pipes=model_pipes[submodel_type]
                            submodel_scores=Submodel_Scores(submodel_datasets, submodel_pipes)
                            model_scores[submodel_type]=submodel_scores
                        return model_scores
                    
                    learning_scores={}
                    match learning_type:
                        case "supervised learning":        
                            model_types=["linear models", "decision trees", "support vector machines", "supervised nearest neighbors", "ensembles", "neural network models"]
                        case "unsupervised learning":
                            model_types=["clustering"]
                    for model_type in model_types:
                        model_datasets=learning_datasets[model_type]
                        model_pipes=learning_pipes[model_type]
                        model_scores=Model_Scores(model_datasets, model_pipes)
                        learning_scores[model_type]=model_scores
                    return learning_scores
                
                
                scores={}
                learning_types=["supervised learning", "unsupervised learning"]
                for learning_type in learning_types:
                    learning_datasets=datasets[learning_type]
                    learning_pipes=pipes[learning_type]
                    learning_scores=Learning_Scores(learning_datasets, learning_pipes)
                    scores[learning_type]=learning_scores
                return scores
            
            def Table(scores):
                """
                
                Takes the scores and returns the table of those scores.

                Parameters
                ----------
                scores : dict
                    The scores of the experiment.

                Returns
                -------
                table : DataFrame
                    The table of the scores.

                """
                
                table=[]
                FEATURE_MATRIX_DDRS_DF=pd.DataFrame(data=list(FEATURE_MATRIX_DDRS), columns=["matrix of features DDR"])
                table.append(FEATURE_MATRIX_DDRS_DF)
                FEATURE_MATRIX_DDR_TYPES=[f"matrix of features DDR == {FEATURE_MATRIX_DDR}" for FEATURE_MATRIX_DDR in FEATURE_MATRIX_DDRS]
                learning_types=["supervised learning", "unsupervised learning"]
                for learning_type in learning_types:
                    learning_scores=scores[learning_type]
                    model_types=[]
                    match learning_type:
                        case "supervised learning":        
                            model_types=["linear models", "decision trees", "support vector machines", "supervised nearest neighbors", "ensembles", "neural network models"]
                        case "unsupervised learning":
                            model_types=["clustering"]
                    for model_type in model_types:
                        model_scores=learning_scores[model_type]
                        submodel_types=[]
                        match model_type:
                            case "linear models":
                                submodel_types=["ordinary least squares", "logistic regression", "polynomial regression"]
                            case "decision trees":
                                submodel_types=["decision tree regressor", "decision tree classifier"]
                            case "support vector machines":
                                submodel_types=["support vector regressor", "support vector classifier"]
                            case "supervised nearest neighbors":
                                submodel_types=["nearest neighbors regressor", "nearest neighbors classifier"]
                            # case "ensembles":
                            #     submodel_types=["random forests"]
                            case "neural network models":
                                submodel_types=["multi-layer perceptrons"]
                            case "clustering":
                                submodel_types=["k means"]
                        for submodel_type in submodel_types:
                            submodel_scores=model_scores[submodel_type]
                            sub2model_types=[]
                            match submodel_type:
                                case "ordinary least squares":
                                    sub2model_types=["ordinary least squares random regression"]
                                case "logistic regression":
                                    sub2model_types=["binary logistic regression"]
                                case "decision tree regressor":
                                    sub2model_types=["decision tree regressor friedman #1"]
                                case "decision tree classifier":
                                    sub2model_types=["decision tree classifier random 2-class classification"]
                                case "support vector regressor":
                                    sub2model_types=["linear support vector regressor"]
                                case "support vector classifier":
                                    sub2model_types=["linear support vector classifier"]
                                case "nearest neighbors regressor":
                                    sub2model_types=["k nearest neighbors regressor"]
                                case "nearest neighbors classifier":
                                    sub2model_types=["k nearest neighbors classifier"]
                                # case "random forests":
                                #     sub2model_types=["random forest regressor"]
                                case "multi-layer perceptrons":
                                    sub2model_types=["multi-layer perceptron regressor", "multi-layer perceptron classifier"]
                                case "k means":
                                    sub2model_types=["k means isotropic Gaussian blobs"]
                                # case "polynomial regression":
                                #     sub2model_types=["quadratic regression"]
                            for sub2model_type in sub2model_types:
                                sub2model_scores=submodel_scores[sub2model_type]
                                sub3model_types=[]
                                match sub2model_type:
                                    case "ordinary least squares random regression":
                                        sub2model_scores_train=[]
                                        sub2model_scores_test=[]
                                        for FEATURE_MATRIX_DDR_TYPE in FEATURE_MATRIX_DDR_TYPES:
                                            sub3model_scores=sub2model_scores[FEATURE_MATRIX_DDR_TYPE]
                                            sub3model_scores_keys=sub3model_scores.keys()
                                            sub3model_scores_train=[]
                                            sub3model_scores_test=[]
                                            for sub3model_scores_key in sub3model_scores_keys:
                                                sub3model_score=sub3model_scores[sub3model_scores_key]
                                                if sub3model_scores_key.split()[-1]=="train":
                                                    sub3model_scores_train.append(sub3model_score)    
                                                else:
                                                    sub3model_scores_test.append(sub3model_score)
                                            sub2model_scores_train.append(sub3model_scores_train)
                                            sub2model_scores_test.append(sub3model_scores_test)
                                        sub2model_scores_train=[np.mean(sub3model_scores_train) for sub3model_scores_train in sub2model_scores_train]
                                        sub2model_scores_train=pd.DataFrame(sub2model_scores_train, columns=[sub2model_type+" train"])
                                        table.append(sub2model_scores_train)
                                        sub2model_scores_test=[np.mean(sub3model_scores_test) for sub3model_scores_test in sub2model_scores_test]
                                        sub2model_scores_test=pd.DataFrame(sub2model_scores_test, columns=[sub2model_type+" test"])
                                        table.append(sub2model_scores_test)
                                    case "binary logistic regression":
                                        sub3model_types=["binary logistic regression random 2-class classification"]
                                    case "decision tree regressor friedman #1":
                                        sub2model_scores_train=[]
                                        sub2model_scores_test=[]
                                        for FEATURE_MATRIX_DDR_TYPE in FEATURE_MATRIX_DDR_TYPES:
                                            sub3model_scores=sub2model_scores[FEATURE_MATRIX_DDR_TYPE]
                                            sub3model_scores_keys=sub3model_scores.keys()
                                            sub3model_scores_train=[]
                                            sub3model_scores_test=[]
                                            for sub3model_scores_key in sub3model_scores_keys:
                                                sub3model_score=sub3model_scores[sub3model_scores_key]
                                                if sub3model_scores_key.split()[-1]=="train":
                                                    sub3model_scores_train.append(sub3model_score)    
                                                else:
                                                    sub3model_scores_test.append(sub3model_score)
                                            sub2model_scores_train.append(sub3model_scores_train)
                                            sub2model_scores_test.append(sub3model_scores_test)
                                        sub2model_scores_train=[np.mean(sub3model_scores_train) for sub3model_scores_train in sub2model_scores_train]
                                        sub2model_scores_train=np.array(sub2model_scores_train)
                                        sub2model_scores_train=pd.DataFrame(sub2model_scores_train, columns=[sub2model_type+" train"])
                                        table.append(sub2model_scores_train)
                                        sub2model_scores_test=[np.mean(sub3model_scores_test) for sub3model_scores_test in sub2model_scores_test]
                                        sub2model_scores_test=pd.DataFrame(sub2model_scores_test, columns=[sub2model_type+" test"])
                                        table.append(sub2model_scores_test) 
                                    case "decision tree classifier random 2-class classification":
                                        sub2model_scores_train=[]
                                        sub2model_scores_test=[]
                                        for FEATURE_MATRIX_DDR_TYPE in FEATURE_MATRIX_DDR_TYPES:
                                            sub3model_scores=sub2model_scores[FEATURE_MATRIX_DDR_TYPE]
                                            sub3model_scores_keys=sub3model_scores.keys()
                                            sub3model_scores_train=[]
                                            sub3model_scores_test=[]
                                            for sub3model_scores_key in sub3model_scores_keys:
                                                sub3model_score=sub3model_scores[sub3model_scores_key]
                                                if sub3model_scores_key.split()[-1]=="train":
                                                    sub3model_scores_train.append(sub3model_score)    
                                                else:
                                                    sub3model_scores_test.append(sub3model_score)
                                            sub2model_scores_train.append(sub3model_scores_train)
                                            sub2model_scores_test.append(sub3model_scores_test)
                                        sub2model_scores_train=[np.mean(sub3model_scores_train) for sub3model_scores_train in sub2model_scores_train]
                                        sub2model_scores_train=pd.DataFrame(sub2model_scores_train, columns=[sub2model_type+" train"])
                                        table.append(sub2model_scores_train)
                                        sub2model_scores_test=[np.mean(sub3model_scores_test) for sub3model_scores_test in sub2model_scores_test]
                                        sub2model_scores_test=pd.DataFrame(sub2model_scores_test, columns=[sub2model_type+" test"])
                                        table.append(sub2model_scores_test)  
                                    case "linear support vector regressor":
                                        sub3model_types=["linear support vector regressor random regression"]
                                    case "linear support vector classifier":
                                        sub3model_types=["linear support vector classifier random 2-class classification"]    
                                    case "k nearest neighbors regressor":
                                        sub3model_types=["k nearest neighbors regressor friedman #1"]
                                    case "k nearest neighbors classifier":
                                        sub3model_types=["k nearest neighbors classifier random 2-class classification"]
                                    # case "random forest regressor":
                                    #     sub3model_types=["random forest regressor friedman #1"]
                                    case "multi-layer perceptron regressor":
                                        sub3model_types=["multi-layer perceptron regressor friedman #1"]
                                    case "multi-layer perceptron classifier":
                                        sub3model_types=["multi-layer perceptron classifier random 2-class classification"]
                                    case "k means isotropic Gaussian blobs":
                                        sub2model_scores_all=[]
                                        for FEATURE_MATRIX_DDR_TYPE in FEATURE_MATRIX_DDR_TYPES:
                                            sub3model_scores=sub2model_scores[FEATURE_MATRIX_DDR_TYPE]
                                            sub3model_scores_keys=sub3model_scores.keys()
                                            sub3model_scores_all=[]
                                            for sub3model_scores_key in sub3model_scores_keys:
                                                sub3model_score=sub3model_scores[sub3model_scores_key]
                                                sub3model_scores_all.append(sub3model_score)
                                            sub2model_scores_all.append(sub3model_scores_train)
                                        sub2model_scores_all=[np.mean(sub3model_scores_all) for sub3model_scores_all in sub2model_scores_all]
                                        sub2model_scores_all=pd.DataFrame(sub2model_scores_all, columns=[sub2model_type])
                                        table.append(sub2model_scores_all)
                                    # case "quadratic regression":
                                    #     sub3model_types=["quadratic regression random quadratic regression"]
                                for sub3model_type in sub3model_types:
                                    sub3model_scores=sub2model_scores[sub3model_type]
                                    match sub3model_type:
                                        case "binary logistic regression random 2-class classification":
                                            sub3model_scores_train=[]
                                            sub3model_scores_test=[]
                                            for FEATURE_MATRIX_DDR_TYPE in FEATURE_MATRIX_DDR_TYPES:
                                                sub4model_scores=sub3model_scores[FEATURE_MATRIX_DDR_TYPE]
                                                sub4model_scores_keys=sub4model_scores.keys()
                                                sub4model_scores_train=[]
                                                sub4model_scores_test=[]
                                                for sub4model_scores_key in sub4model_scores_keys:
                                                    sub4model_score=sub4model_scores[sub4model_scores_key]
                                                    if sub4model_scores_key.split()[-1]=="train":
                                                        sub4model_scores_train.append(sub4model_score)    
                                                    else:
                                                        sub4model_scores_test.append(sub4model_score)
                                                sub3model_scores_train.append(sub4model_scores_train)
                                                sub3model_scores_test.append(sub4model_scores_test)
                                            sub3model_scores_train=[np.mean(sub4model_scores_train) for sub4model_scores_train in sub3model_scores_train]
                                            sub3model_scores_train=pd.DataFrame(sub3model_scores_train, columns=[sub3model_type+" train"])
                                            table.append(sub3model_scores_train)
                                            sub3model_scores_test=[np.mean(sub4model_scores_test) for sub4model_scores_test in sub3model_scores_test]
                                            sub3model_scores_test=pd.DataFrame(sub3model_scores_test, columns=[sub3model_type+" test"])
                                            table.append(sub3model_scores_test)
                                        case "linear support vector regressor random regression":
                                            sub3model_scores_train=[]
                                            sub3model_scores_test=[]
                                            for FEATURE_MATRIX_DDR_TYPE in FEATURE_MATRIX_DDR_TYPES:
                                                sub4model_scores=sub3model_scores[FEATURE_MATRIX_DDR_TYPE]
                                                sub4model_scores_keys=sub4model_scores.keys()
                                                sub4model_scores_train=[]
                                                sub4model_scores_test=[]
                                                for sub4model_scores_key in sub4model_scores_keys:
                                                    sub4model_score=sub4model_scores[sub4model_scores_key]
                                                    if sub4model_scores_key.split()[-1]=="train":
                                                        sub4model_scores_train.append(sub4model_score)    
                                                    else:
                                                        sub4model_scores_test.append(sub4model_score)
                                                sub3model_scores_train.append(sub4model_scores_train)
                                                sub3model_scores_test.append(sub4model_scores_test)
                                            sub3model_scores_train=[np.mean(sub4model_scores_train) for sub4model_scores_train in sub3model_scores_train]
                                            sub3model_scores_train=pd.DataFrame(sub3model_scores_train, columns=[sub3model_type+" train"])
                                            table.append(sub3model_scores_train)
                                            sub3model_scores_test=[np.mean(sub4model_scores_test) for sub4model_scores_test in sub3model_scores_test]
                                            sub3model_scores_test=pd.DataFrame(sub3model_scores_test, columns=[sub3model_type+" test"])
                                            table.append(sub3model_scores_test)
                                        case "linear support vector classifier random 2-class classification":
                                            sub3model_scores_train=[]
                                            sub3model_scores_test=[]
                                            for FEATURE_MATRIX_DDR_TYPE in FEATURE_MATRIX_DDR_TYPES:
                                                sub4model_scores=sub3model_scores[FEATURE_MATRIX_DDR_TYPE]
                                                sub4model_scores_keys=sub4model_scores.keys()
                                                sub4model_scores_train=[]
                                                sub4model_scores_test=[]
                                                for sub4model_scores_key in sub4model_scores_keys:
                                                    sub4model_score=sub4model_scores[sub4model_scores_key]
                                                    if sub4model_scores_key.split()[-1]=="train":
                                                        sub4model_scores_train.append(sub4model_score)    
                                                    else:
                                                        sub4model_scores_test.append(sub4model_score)
                                                sub3model_scores_train.append(sub4model_scores_train)
                                                sub3model_scores_test.append(sub4model_scores_test)
                                            sub3model_scores_train=[np.mean(sub4model_scores_train) for sub4model_scores_train in sub3model_scores_train]
                                            sub3model_scores_train=pd.DataFrame(sub3model_scores_train, columns=[sub3model_type+" train"])
                                            table.append(sub3model_scores_train)
                                            sub3model_scores_test=[np.mean(sub4model_scores_test) for sub4model_scores_test in sub3model_scores_test]
                                            sub3model_scores_test=pd.DataFrame(sub3model_scores_test, columns=[sub3model_type+" test"])
                                            table.append(sub3model_scores_test)           
                                        case "k nearest neighbors regressor friedman #1":
                                            sub3model_scores_train=[]
                                            sub3model_scores_test=[]
                                            for FEATURE_MATRIX_DDR_TYPE in FEATURE_MATRIX_DDR_TYPES:
                                                sub4model_scores=sub3model_scores[FEATURE_MATRIX_DDR_TYPE]
                                                sub4model_scores_keys=sub4model_scores.keys()
                                                sub4model_scores_train=[]
                                                sub4model_scores_test=[]
                                                for sub4model_scores_key in sub4model_scores_keys:
                                                    sub4model_score=sub4model_scores[sub4model_scores_key]
                                                    if sub4model_scores_key.split()[-1]=="train":
                                                        sub4model_scores_train.append(sub4model_score)    
                                                    else:
                                                        sub4model_scores_test.append(sub4model_score)
                                                sub3model_scores_train.append(sub4model_scores_train)
                                                sub3model_scores_test.append(sub4model_scores_test)
                                            sub3model_scores_train=[np.mean(sub4model_scores_train) for sub4model_scores_train in sub3model_scores_train]
                                            sub3model_scores_train=pd.DataFrame(sub3model_scores_train, columns=[sub3model_type+" train"])
                                            table.append(sub3model_scores_train)
                                            sub3model_scores_test=[np.mean(sub4model_scores_test) for sub4model_scores_test in sub3model_scores_test]
                                            sub3model_scores_test=pd.DataFrame(sub3model_scores_test, columns=[sub3model_type+" test"])
                                            table.append(sub3model_scores_test)
                                        case "k nearest neighbors classifier random 2-class classification":
                                            sub3model_scores_train=[]
                                            sub3model_scores_test=[]
                                            for FEATURE_MATRIX_DDR_TYPE in FEATURE_MATRIX_DDR_TYPES:
                                                sub4model_scores=sub3model_scores[FEATURE_MATRIX_DDR_TYPE]
                                                sub4model_scores_keys=sub4model_scores.keys()
                                                sub4model_scores_train=[]
                                                sub4model_scores_test=[]
                                                for sub4model_scores_key in sub4model_scores_keys:
                                                    sub4model_score=sub4model_scores[sub4model_scores_key]
                                                    if sub4model_scores_key.split()[-1]=="train":
                                                        sub4model_scores_train.append(sub4model_score)    
                                                    else:
                                                        sub4model_scores_test.append(sub4model_score)
                                                sub3model_scores_train.append(sub4model_scores_train)
                                                sub3model_scores_test.append(sub4model_scores_test)
                                            sub3model_scores_train=[np.mean(sub4model_scores_train) for sub4model_scores_train in sub3model_scores_train]
                                            sub3model_scores_train=pd.DataFrame(sub3model_scores_train, columns=[sub3model_type+" train"])
                                            table.append(sub3model_scores_train)
                                            sub3model_scores_test=[np.mean(sub4model_scores_test) for sub4model_scores_test in sub3model_scores_test]
                                            sub3model_scores_test=pd.DataFrame(sub3model_scores_test, columns=[sub3model_type+" test"])
                                            table.append(sub3model_scores_test)      
                                        # case "random forest regressor friedman #1":
                                        #     sub3model_scores_train=[]
                                        #     sub3model_scores_test=[]
                                        #     for FEATURE_MATRIX_DDR_TYPE in FEATURE_MATRIX_DDR_TYPES:
                                        #         sub4model_scores=sub3model_scores[FEATURE_MATRIX_DDR_TYPE]
                                        #         sub4model_scores_keys=sub4model_scores.keys()
                                        #         sub4model_scores_train=[]
                                        #         sub4model_scores_test=[]
                                        #         for sub4model_scores_key in sub4model_scores_keys:
                                        #             sub4model_score=sub4model_scores[sub4model_scores_key]
                                        #             if sub4model_scores_key.split()[-1]=="train":
                                        #                 sub4model_scores_train.append(sub4model_score)    
                                        #             else:
                                        #                 sub4model_scores_test.append(sub4model_score)
                                        #         sub3model_scores_train.append(sub4model_scores_train)
                                        #         sub3model_scores_test.append(sub4model_scores_test)
                                        #     sub3model_scores_train=[np.mean(sub4model_scores_train) for sub4model_scores_train in sub3model_scores_train]
                                        #     sub3model_scores_train=np.array(sub3model_scores_train)
                                        #     table.append(sub3model_scores_train)
                                        #     sub3model_scores_test=[np.mean(sub4model_scores_test) for sub4model_scores_test in sub3model_scores_test]
                                        #     sub3model_scores_test=np.array(sub3model_scores_test)
                                        #     table.append(sub3model_scores_test)
                                        case "multi-layer perceptron regressor random regression":
                                            sub3model_scores_train=[]
                                            sub3model_scores_test=[]
                                            for FEATURE_MATRIX_DDR_TYPE in FEATURE_MATRIX_DDR_TYPES:
                                                sub4model_scores=sub3model_scores[FEATURE_MATRIX_DDR_TYPE]
                                                sub4model_scores_keys=sub4model_scores.keys()
                                                sub4model_scores_train=[]
                                                sub4model_scores_test=[]
                                                for sub4model_scores_key in sub4model_scores_keys:
                                                    sub4model_score=sub4model_scores[sub4model_scores_key]
                                                    if sub4model_scores_key.split()[-1]=="train":
                                                        sub4model_scores_train.append(sub4model_score)    
                                                    else:
                                                        sub4model_scores_test.append(sub4model_score)
                                                sub3model_scores_train.append(sub4model_scores_train)
                                                sub3model_scores_test.append(sub4model_scores_test)
                                            sub3model_scores_train=[np.mean(sub4model_scores_train) for sub4model_scores_train in sub3model_scores_train]
                                            sub3model_scores_train=pd.DataFrame(sub3model_scores_train, columns=[sub3model_type+" train"])
                                            table.append(sub3model_scores_train)
                                            sub3model_scores_test=[np.mean(sub4model_scores_test) for sub4model_scores_test in sub3model_scores_test]
                                            sub3model_scores_test=pd.DataFrame(sub3model_scores_test, columns=[sub3model_type+" test"])
                                            table.append(sub3model_scores_test)
                                        case "multi-layer perceptron classifier random 2-class classification":
                                            sub3model_scores_train=[]
                                            sub3model_scores_test=[]
                                            for FEATURE_MATRIX_DDR_TYPE in FEATURE_MATRIX_DDR_TYPES:
                                                sub4model_scores=sub3model_scores[FEATURE_MATRIX_DDR_TYPE]
                                                sub4model_scores_keys=sub4model_scores.keys()
                                                sub4model_scores_train=[]
                                                sub4model_scores_test=[]
                                                for sub4model_scores_key in sub4model_scores_keys:
                                                    sub4model_score=sub4model_scores[sub4model_scores_key]
                                                    if sub4model_scores_key.split()[-1]=="train":
                                                        sub4model_scores_train.append(sub4model_score)    
                                                    else:
                                                        sub4model_scores_test.append(sub4model_score)
                                                sub3model_scores_train.append(sub4model_scores_train)
                                                sub3model_scores_test.append(sub4model_scores_test)
                                            sub3model_scores_train=[np.mean(sub4model_scores_train) for sub4model_scores_train in sub3model_scores_train]
                                            sub3model_scores_train=pd.DataFrame(sub3model_scores_train, columns=[sub3model_type+" train"])
                                            table.append(sub3model_scores_train)
                                            sub3model_scores_test=[np.mean(sub4model_scores_test) for sub4model_scores_test in sub3model_scores_test]
                                            sub3model_scores_test=pd.DataFrame(sub3model_scores_test, columns=[sub3model_type+" test"])
                                            table.append(sub3model_scores_test)      
                table=pd.concat(table, axis=1)
                return table
            
            # def Plots(parameters, datasets, pipes, scores):
            #     """
            #     Takes the parameters, datasets, pipes, and scores and returns the plots.
    
            #     Parameters
            #     ----------
            #     parameters : Dict
            #         Parameters of the experiment.
            #     scores : Dict
            #         Scores from the experiment.
    
            #     Returns
            #     -------
            #     plots : Dict
            #         Plots from the experiment.
    
            #     """
                
            #     def Dataset_Plots(parameters, datasets, pipes):
            #         """
            #         Takes the parameters, datasets, and pipes and returns the dataset plots
    
            #         Parameters
            #         ----------
            #         parameters : Dict
            #             Parameters of the experiment.
            #         datasets : Dict
            #             Datasets of the experiment.
            #         pipes : Dict
            #             Pipes of the experiment.
    
            #         Returns
            #         -------
            #         dataset_plots: Dict
            #             Dataset plots from the experiment.
    
            #         """
                
            #         def Learning_Dataset_Plots(learning_parameters, learning_datasets, learning_pipes):
            #             """
            #             Takes the learning parameters, learning datasets, and learning pipes and returns the learning dataset plots.
        
            #             Parameters
            #             ----------
            #             learning_parameters : Dict
            #                 Parameters of the learning type.
            #             learning_datasets : Dict
            #                 Datasets from the learning type.
            #             learning_pipes : Dict
            #                 Pipes from the learning type.
        
            #             Returns
            #             -------
            #             learning_dataset_plots : Dict
            #                 Dataset plots from the learning type.
        
            #             """
                        
            #             def Model_Dataset_Plots(model_parameters, model_datasets, model_pipes):
            #                 """
            #                 Takes the model parameters, model datasets, and model pipes and returns the model dataset plots.
        
            #                 Parameters
            #                 ----------
            #                 model_parameters : Dict
            #                     Parameters of the model type.
            #                 model_datasets : Dict
            #                     Datasets from the model type.
            #                 model_pipes : Dict
            #                     Pipes from the model type.
        
            #                 Returns
            #                 -------
            #                 model_dataset_plots : Dict
            #                     Dataset plots from the model type.
        
            #                 """
                            
            #                 def Submodel_Dataset_Plots(submodel_parameters, submodel_datasets, submodel_pipes):
            #                     """
            #                     Takes the submodel parameters, submodel, datasets, and submodel pipes and returns the submodel dataset plots.
        
            #                     Parameters
            #                     ----------
            #                     submodel_parameters : Dict
            #                         Parameters of the submodel type.
            #                     submodel_datasets : Dict
            #                         Datasets from the submodel type.
            #                     submodel_pipes : Dict
            #                         Pipes from the submodel type.
        
            #                     Returns
            #                     -------
            #                     submodel_dataset_plots : Dict
            #                         Dataset plots from the submodel type.
        
            #                     """
                            
            #                     def Sub2model_Dataset_Plots(sub2model_parameters, sub2model_datasets, sub2model_pipes):
            #                         """
            #                         Takes the sub2model parameters, sub2model datasets, and sub2model pipes and returns the sub2model plots.
        
            #                         Parameters
            #                         ----------
            #                         sub2model_parameters : Dict
            #                             Parameters of the sub2model type.
            #                         sub2model_datasets : Dict
            #                             Datasets from the sub2model type.
            #                         sub2model_pipes : Dict
            #                             Pipes from the sub2model type.
        
            #                         Returns
            #                         -------
            #                         sub2model_dataset_plots : Dict
            #                             Dataset plots from the sub2model type.
        
            #                         """
                                    
            #                         def Sub3model_Dataset_Plots(sub3model_parameters, sub3model_datasets, sub3model_pipes):
            #                             """
            #                             Takes the sub3model parameters, sub3model datasets, and sub3model pipes and returns the sub3model plots.
        
            #                             Parameters
            #                             ----------
            #                             sub3model_parameters : Dict
            #                                 Parameters of the sub3model type.
            #                             sub3model_datasets : Dict
            #                                 Datasets from the sub3model type.
            #                             sub3model_pipes : Dict
            #                                 Pipes from the sub3model type.
        
            #                             Returns
            #                             -------
            #                             sub3model_dataset_plots : Dict
            #                                 Dataset plots from the sub3model type.
        
            #                             """
                                        
            #                             sub3model_dataset_plots={}
            #                             sub3model_types=[]
            #                             sub4model_types=sub3model_parameters.keys()
            #                             feature_matrix_ddrs=[]
            #                             match sub3model_type:
            #                                 case "linear support vector regressor random regression":
            #                                     feature_matrix_test=sub3model_datasets["testing matrix of features"]
            #                                     targets_test=sub3model_datasets["testing targets"]
            #                                     feature_matrix_train=sub3model_datasets["training matrix of features"]
            #                                     targets_train=sub3model_datasets["training targets"]
            #                                     sub4model_pipe_types=sub3model_pipes.keys()
            #                                     for sub4model_pipe_type in tqdm.tqdm(sub4model_pipe_types,desc=f"Getting scores for {sub3model_type}"):
            #                                         sub4model_pipes=sub3model_pipes[sub4model_pipe_type]
            #                                         sub5model_pipe_types=sub4model_pipes.keys()
            #                                         for sub5model_pipe_type in sub5model_pipe_types:
            #                                             sub5model_pipe=sub4model_pipes[sub5model_pipe_type]
            #                                             scaler_pipe=sub5model_pipe[:1]
            #                                             feature_matrix_train_plot=feature_matrix_train
            #                                             feature_matrix_train_plot_scaled=scaler_pipe.fit_transform(feature_matrix_train_plot)
            #                                             feature_matrix_test_plot=feature_matrix_test
            #                                             feature_matrix_test_plot_scaled=scaler_pipe.fit_transform(feature_matrix_test_plot)
            #                                             data_train_plot=np.concatenate((feature_matrix_train_plot_scaled, targets_train.reshape(-1,1)), axis=1)
            #                                             df_train=pd.DataFrame(data_train_plot, columns=["x", "y"])
            #                                             data_test_plot=np.concatenate((feature_matrix_test_plot_scaled, targets_test.reshape(-1,1)), axis=1)
            #                                             df_test=pd.DataFrame(data_test_plot, columns=["x", "y"])
            #                                             grid_train=sns.pairplot(data=df_train)
            #                                             grid_train_fig=grid_train.figure
            #                                             grid_train_fig.suptitle(f"{sub3model_type} training data with DDR-Invariant Standardized Features")
            #                                             plt.show(grid_train_fig)
            #                                             #plt.close(grid_train_fig)
            #                                             grid_test=sns.pairplot(data=df_test)
            #                                             grid_test_fig=grid_test.figure
            #                                             grid_test_fig.suptitle(f"{sub3model_type} testing data with DDR-Invariant Standardized Features")
            #                                             plt.show(grid_test_fig)
            #                                             #plt.close(grid_test_fig)
            #                                             sub3model_dataset_plots[f"DDR-invariant feature standardized {sub3model_type} training data"]=grid_train_fig
            #                                             sub3model_dataset_plots[f"DDR-invariant feature standardized {sub3model_type} testing data"]=grid_test_fig
            #                                 case "k nearest neighbors regressor friedman #1":
            #                                     feature_matrix_test=sub3model_datasets["testing matrix of features"]
            #                                     targets_test=sub3model_datasets["testing targets"]
            #                                     feature_matrix_train=sub3model_datasets["training matrix of features"]
            #                                     targets_train=sub3model_datasets["training targets"]
            #                                     sub4model_pipe_types=sub3model_pipes.keys()
            #                                     for sub4model_pipe_type in tqdm.tqdm(sub4model_pipe_types,desc=f"Getting scores for {sub3model_type}"):
            #                                         sub4model_pipes=sub3model_pipes[sub4model_pipe_type]
            #                                         sub5model_pipe_types=sub4model_pipes.keys()
            #                                         for sub5model_pipe_type in sub5model_pipe_types:
            #                                             sub5model_pipe=sub4model_pipes[sub5model_pipe_type]
            #                                             scaler_pipe=sub5model_pipe[:1]
            #                                             feature_matrix_train_plot=feature_matrix_train
            #                                             feature_matrix_train_plot_scaled=scaler_pipe.fit_transform(feature_matrix_train_plot)
            #                                             feature_matrix_test_plot=feature_matrix_test
            #                                             feature_matrix_test_plot_scaled=scaler_pipe.fit_transform(feature_matrix_test_plot)
            #                                             data_train_plot=np.concatenate((feature_matrix_train_plot_scaled, targets_train.reshape(-1,1)), axis=1)
            #                                             df_train=pd.DataFrame(data_train_plot, columns=["X[0]", "X[1]", "X[2]", "X[3]", "X[4]", "y"])
            #                                             data_test_plot=np.concatenate((feature_matrix_test_plot_scaled, targets_test.reshape(-1,1)), axis=1)
            #                                             df_test=pd.DataFrame(data_test_plot, columns=["X[0]", "X[1]", "X[2]", "X[3]", "X[4]", "y"])
            #                                             grid_train=sns.pairplot(data=df_train)
            #                                             grid_test=sns.pairplot(data=df_test)
            #                                             grid_train_fig=grid_train.figure
            #                                             grid_train_fig.suptitle(f"{sub3model_type} training data with DDR-Invariant Standardized Features")
            #                                             grid_test_fig=grid_test.figure
            #                                             grid_test_fig.suptitle(f"{sub3model_type} testing data with DDR-Invariant Standardized Features")
            #                                             plt.show(grid_train_fig)
            #                                             #plt.close(grid_train_fig)
            #                                             plt.show(grid_test_fig)
            #                                             #plt.close(grid_test_fig)
            #                                             sub3model_dataset_plots[f"DDR-invariant feature standardized {sub3model_type} training data"]=grid_train_fig
            #                                             sub3model_dataset_plots[f"DDR-invariant feature standardized {sub3model_type} testing data"]=grid_test_fig
            #                                 case "random forest regressor friedman #1":
            #                                     feature_matrix_test=sub3model_datasets["testing matrix of features"]
            #                                     targets_test=sub3model_datasets["testing targets"]
            #                                     feature_matrix_train=sub3model_datasets["training matrix of features"]
            #                                     targets_train=sub3model_datasets["training targets"]
            #                                     sub4model_pipe_types=sub3model_pipes.keys()
            #                                     for sub4model_pipe_type in tqdm.tqdm(sub4model_pipe_types,desc=f"Getting scores for {sub3model_type}"):
            #                                         sub4model_pipes=sub3model_pipes[sub4model_pipe_type]
            #                                         sub5model_pipe_types=sub4model_pipes.keys()
            #                                         for sub5model_pipe_type in sub5model_pipe_types:
            #                                             sub5model_pipe=sub4model_pipes[sub5model_pipe_type]
            #                                             scaler_pipe=sub5model_pipe[:1]
            #                                             feature_matrix_train_plot=feature_matrix_train
            #                                             feature_matrix_train_plot_scaled=scaler_pipe.fit_transform(feature_matrix_train_plot)
            #                                             feature_matrix_test_plot=feature_matrix_test
            #                                             feature_matrix_test_plot_scaled=scaler_pipe.fit_transform(feature_matrix_test_plot)
            #                                             data_train_plot=np.concatenate((feature_matrix_train_plot_scaled, targets_train.reshape(-1,1)), axis=1)
            #                                             df_train=pd.DataFrame(data_train_plot, columns=["X[0]", "X[1]", "X[2]", "X[3]", "X[4]", "y"])
            #                                             data_test_plot=np.concatenate((feature_matrix_test_plot_scaled, targets_test.reshape(-1,1)), axis=1)
            #                                             df_test=pd.DataFrame(data_test_plot, columns=["X[0]", "X[1]", "X[2]", "X[3]", "X[4]", "y"])
            #                                             grid_train=sns.pairplot(data=df_train)
            #                                             grid_test=sns.pairplot(data=df_test)
            #                                             grid_train_fig=grid_train.figure
            #                                             grid_train_fig.suptitle(f"{sub3model_type} training data with DDR-Invariant Standardized Features")
            #                                             grid_test_fig=grid_test.figure
            #                                             grid_test_fig.suptitle(f"{sub3model_type} testing data with DDR-Invariant Standardized Features")
            #                                             plt.show(grid_train_fig)
            #                                             #plt.close(grid_train_fig)
            #                                             plt.show(grid_test_fig)
            #                                             #plt.close(grid_test_fig)
            #                                             sub3model_dataset_plots[f"DDR-invariant feature standardized {sub3model_type} training data"]=grid_train_fig
            #                                             sub3model_dataset_plots[f"DDR-invariant feature standardized {sub3model_type} testing data"]=grid_test_fig
            #                                 case "multi-layer perceptron regressor friedman #1":
            #                                     feature_matrix_test=sub3model_datasets["testing matrix of features"]
            #                                     targets_test=sub3model_datasets["testing targets"]
            #                                     feature_matrix_train=sub3model_datasets["training matrix of features"]
            #                                     targets_train=sub3model_datasets["training targets"]
            #                                     sub4model_pipe_types=sub3model_pipes.keys()
            #                                     for sub4model_pipe_type in tqdm.tqdm(sub4model_pipe_types,desc=f"Getting scores for {sub3model_type}"):
            #                                         sub4model_pipes=sub3model_pipes[sub4model_pipe_type]
            #                                         sub5model_pipe_types=sub4model_pipes.keys()
            #                                         for sub5model_pipe_type in sub5model_pipe_types:
            #                                             sub5model_pipe=sub4model_pipes[sub5model_pipe_type]
            #                                             scaler_pipe=sub5model_pipe[:1]
            #                                             feature_matrix_train_plot=feature_matrix_train
            #                                             feature_matrix_train_plot_scaled=scaler_pipe.fit_transform(feature_matrix_train_plot)
            #                                             feature_matrix_test_plot=feature_matrix_test
            #                                             feature_matrix_test_plot_scaled=scaler_pipe.fit_transform(feature_matrix_test_plot)
            #                                             data_train_plot=np.concatenate((feature_matrix_train_plot_scaled, targets_train.reshape(-1,1)), axis=1)
            #                                             df_train=pd.DataFrame(data_train_plot, columns=["X[0]", "X[1]", "X[2]", "X[3]", "X[4]", "y"])
            #                                             data_test_plot=np.concatenate((feature_matrix_test_plot_scaled, targets_test.reshape(-1,1)), axis=1)
            #                                             df_test=pd.DataFrame(data_test_plot, columns=["X[0]", "X[1]", "X[2]", "X[3]", "X[4]", "y"])
            #                                             grid_train=sns.pairplot(data=df_train)
            #                                             grid_test=sns.pairplot(data=df_test)
            #                                             grid_train_fig=grid_train.figure
            #                                             grid_train_fig.suptitle(f"{sub3model_type} training data with DDR-Invariant Standardized Features")
            #                                             grid_test_fig=grid_test.figure
            #                                             grid_test_fig.suptitle(f"{sub3model_type} testing data with DDR-Invariant Standardized Features")
            #                                             plt.show(grid_train_fig)
            #                                             #plt.close(grid_train_fig)
            #                                             plt.show(grid_test_fig)
            #                                             #plt.close(grid_test_fig)
            #                                             sub3model_dataset_plots[f"DDR-invariant feature standardized {sub3model_type} training data"]=grid_train_fig
            #                                             sub3model_dataset_plots[f"DDR-invariant feature standardized {sub3model_type} testing data"]=grid_test_fig
            #                             return sub2model_dataset_plots
                                    
            #                         sub2model_dataset_plots={}
            #                         sub3model_types=[]
            #                         feature_matrix_ddrs=[]
            #                         match sub2model_type:
            #                             case "ordinary least squares random regression":
            #                                 feature_matrix_test=sub2model_datasets["testing matrix of features"]
            #                                 targets_test=sub2model_datasets["testing targets"]
            #                                 feature_matrix_train=sub2model_datasets["training matrix of features"]
            #                                 targets_train=sub2model_datasets["training targets"]
            #                                 sub3model_pipe_types=sub2model_pipes.keys()
            #                                 for sub3model_pipe_type in tqdm.tqdm(sub3model_pipe_types,desc=f"Getting scores for {sub2model_type}"):
            #                                     sub3model_pipes=sub2model_pipes[sub3model_pipe_type]
            #                                     sub4model_pipe_types=sub3model_pipes.keys()
            #                                     for sub4model_pipe_type in sub4model_pipe_types:
            #                                         sub4model_pipe=sub3model_pipes[sub4model_pipe_type]
            #                                         scaler_pipe=sub4model_pipe[:1]
            #                                         feature_matrix_train_plot=feature_matrix_train
            #                                         feature_matrix_train_plot_scaled=scaler_pipe.fit_transform(feature_matrix_train_plot)
            #                                         feature_matrix_test_plot=feature_matrix_test
            #                                         feature_matrix_test_plot_scaled=scaler_pipe.fit_transform(feature_matrix_test_plot)
            #                                         data_train_plot=np.concatenate((feature_matrix_train_plot_scaled, targets_train.reshape(-1,1)), axis=1)
            #                                         df_train=pd.DataFrame(data_train_plot, columns=["x", "y"])
            #                                         data_test_plot=np.concatenate((feature_matrix_test_plot_scaled, targets_test.reshape(-1,1)), axis=1)
            #                                         df_test=pd.DataFrame(data_test_plot, columns=["x", "y"])
            #                                         grid_train=sns.pairplot(data=df_train)
            #                                         grid_test=sns.pairplot(data=df_test)
            #                                         grid_train_fig=grid_train.figure
            #                                         grid_train_fig.suptitle(f"{sub2model_type} training data with DDR-Invariant Standardized Features")
            #                                         grid_test_fig=grid_test.figure
            #                                         grid_test_fig.suptitle(f"{sub2model_type} testing data with DDR-Invariant Standardized Features")
            #                                         plt.show(grid_train_fig)
            #                                         #plt.close(grid_train_fig)
            #                                         plt.show(grid_test_fig)
            #                                         #plt.close(grid_test_fig)
            #                                         sub2model_dataset_plots[f"DDR-invariant feature standardized {sub2model_type} training data"]=grid_train_fig
            #                                         sub2model_dataset_plots[f"DDR-invariant feature standardized {sub2model_type} testing data"]=grid_test_fig
            #                             case "decision tree regressor friedman #1":
            #                                 feature_matrix_test=sub2model_datasets["testing matrix of features"]
            #                                 targets_test=sub2model_datasets["testing targets"]
            #                                 feature_matrix_train=sub2model_datasets["training matrix of features"]
            #                                 targets_train=sub2model_datasets["training targets"]
            #                                 sub3model_pipe_types=sub2model_pipes.keys()
            #                                 for sub3model_pipe_type in tqdm.tqdm(sub3model_pipe_types,desc=f"Getting scores for {sub2model_type}"):
            #                                     sub3model_pipes=sub2model_pipes[sub3model_pipe_type]
            #                                     sub4model_pipe_types=sub3model_pipes.keys()
            #                                     for sub4model_pipe_type in sub4model_pipe_types:
            #                                         sub4model_pipe=sub3model_pipes[sub4model_pipe_type]
            #                                         scaler_pipe=sub4model_pipe[:1]
            #                                         feature_matrix_train_plot=feature_matrix_train
            #                                         feature_matrix_train_plot_scaled=scaler_pipe.fit_transform(feature_matrix_train_plot)
            #                                         feature_matrix_test_plot=feature_matrix_test
            #                                         feature_matrix_test_plot_scaled=scaler_pipe.fit_transform(feature_matrix_test_plot)
            #                                         data_train_plot=np.concatenate((feature_matrix_train_plot_scaled, targets_train.reshape(-1,1)), axis=1)
            #                                         df_train=pd.DataFrame(data_train_plot, columns=["X[0]", "X[1]", "X[2]", "X[3]", "X[4]", "y"])
            #                                         data_test_plot=np.concatenate((feature_matrix_test_plot_scaled, targets_test.reshape(-1,1)), axis=1)
            #                                         df_test=pd.DataFrame(data_test_plot, columns=["X[0]", "X[1]", "X[2]", "X[3]", "X[4]", "y"])
            #                                         grid_train=sns.pairplot(data=df_train)
            #                                         grid_test=sns.pairplot(data=df_test)
            #                                         grid_train_fig=grid_train.figure
            #                                         grid_train_fig.suptitle(f"{sub2model_type} training data with DDR-Invariant Standardized Features")
            #                                         grid_test_fig=grid_test.figure
            #                                         grid_test_fig.suptitle(f"{sub2model_type} testing data with DDR-Invariant Standardized Features")
            #                                         plt.show(grid_train_fig)
            #                                         #plt.close(grid_train_fig)
            #                                         plt.show(grid_test_fig)
            #                                         #plt.close(grid_test_fig)
            #                                         sub2model_dataset_plots[f"DDR-invariant feature standardized {sub2model_type} training data"]=grid_train_fig
            #                                         sub2model_dataset_plots[f"DDR-invariant feature standardized {sub2model_type} testing data"]=grid_test_fig
            #                             case "decision tree classifier random 2-class classification":
            #                                 feature_matrix_test=sub2model_datasets["testing matrix of features"]
            #                                 targets_test=sub2model_datasets["testing targets"]
            #                                 feature_matrix_train=sub2model_datasets["training matrix of features"]
            #                                 targets_train=sub2model_datasets["training targets"]
            #                                 sub3model_pipe_types=sub2model_pipes.keys()
            #                                 for sub3model_pipe_type in tqdm.tqdm(sub3model_pipe_types,desc=f"Getting scores for {sub2model_type}"):
            #                                     sub3model_pipes=sub2model_pipes[sub3model_pipe_type]
            #                                     sub4model_pipe_types=sub3model_pipes.keys()
            #                                     for sub4model_pipe_type in sub4model_pipe_types:
            #                                         sub4model_pipe=sub3model_pipes[sub4model_pipe_type]
            #                                         scaler_pipe=sub4model_pipe[:1]
            #                                         feature_matrix_train_plot=feature_matrix_train
            #                                         feature_matrix_train_plot_scaled=scaler_pipe.fit_transform(feature_matrix_train_plot)
            #                                         feature_matrix_test_plot=feature_matrix_test
            #                                         feature_matrix_test_plot_scaled=scaler_pipe.fit_transform(feature_matrix_test_plot)
            #                                         data_train_plot=np.concatenate((feature_matrix_train_plot_scaled, targets_train.reshape(-1,1)), axis=1)
            #                                         df_train=pd.DataFrame(data_train_plot, columns=["x", "y"])
            #                                         data_test_plot=np.concatenate((feature_matrix_test_plot_scaled, targets_test.reshape(-1,1)), axis=1)
            #                                         df_test=pd.DataFrame(data_test_plot, columns=["x", "y"])
            #                                         grid_train=sns.pairplot(data=df_train)
            #                                         grid_test=sns.pairplot(data=df_test)
            #                                         grid_train_fig=grid_train.figure
            #                                         grid_train_fig.suptitle(f"{sub2model_type} training data with DDR-Invariant Standardized Features")
            #                                         grid_test_fig=grid_test.figure
            #                                         grid_test_fig.suptitle(f"{sub2model_type} testing data with DDR-Invariant Standardized Features")
            #                                         plt.show(grid_train_fig)
            #                                         #plt.close(grid_train_fig)
            #                                         plt.show(grid_test_fig)
            #                                         #plt.close(grid_test_fig)
            #                                         sub2model_dataset_plots[f"DDR-invariant feature standardized {sub2model_type} training data"]=grid_train_fig
            #                                         sub2model_dataset_plots[f"DDR-invariant feature standardized {sub2model_type} testing data"]=grid_test_fig
            #                             case "linear support vector regressor":
            #                                 sub3model_types=["linear support vector regressor random regression"]
            #                             case "k nearest neighbors regressor":
            #                                 sub3model_types=["k nearest neighbors regressor friedman #1"]
            #                             case "random forest regressor":
            #                                 sub3model_types=["random forest regressor friedman #1"]
            #                             case "multi-layer perceptron regressor":
            #                                 sub3model_types=["multi-layer perceptron regressor friedman #1"]
            #                             case "k means isotropic Gaussian blobs":
            #                                 feature_matrix=sub2model_datasets["matrix of features"]
            #                                 targets=sub2model_datasets["targets"]
            #                                 sub3model_pipe_types=sub2model_pipes.keys()
            #                                 for sub3model_pipe_type in tqdm.tqdm(sub3model_pipe_types,desc=f"Getting scores for {sub2model_type}"):
            #                                     sub3model_pipes=sub2model_pipes[sub3model_pipe_type]
            #                                     sub4model_pipe_types=sub3model_pipes.keys()
            #                                     for sub4model_pipe_type in sub4model_pipe_types:
            #                                         sub4model_pipe=sub3model_pipes[sub4model_pipe_type]
            #                                         scaler_pipe=sub4model_pipe[:1]
            #                                         feature_matrix_plot=feature_matrix
            #                                         feature_matrix_plot_scaled=scaler_pipe.fit_transform(feature_matrix_plot)
            #                                         data_plot=np.concatenate((feature_matrix_plot_scaled, targets.reshape(-1,1)), axis=1)
            #                                         df=pd.DataFrame(data_plot, columns=["x", "y"])
            #                                         grid=sns.pairplot(data=df)
            #                                         grid_fig=grid.figure
            #                                         grid_fig.suptitle(f"{sub2model_type} data with DDR-Invariant Standardized Features")
            #                                         plt.show(grid_fig)
            #                                         #plt.close(grid_fig)
            #                                         sub2model_dataset_plots[f"DDR-invariant feature standardized {sub2model_type} data"]=grid_fig
            #                             # case "quadratic regression":
            #                             #     sub3model_types=["quadratic regression random quadratic regression"]
            #                         for sub3model_type in sub3model_types:
            #                             sub3model_parameters=sub2model_parameters[sub3model_type]
            #                             sub3model_datasets=sub2model_datasets[sub3model_type]
            #                             sub3model_pipes=sub2model_pipes[sub3model_type]
            #                             sub3model_dataset_plots=Sub3model_Dataset_Plots(sub3model_parameters, sub3model_datasets, sub3model_pipes)
            #                             sub2model_dataset_plots[sub3model_type]=sub3model_dataset_plots
            #                         return sub2model_dataset_plots
                            
            #                     submodel_dataset_plots={}
            #                     sub2model_types=[]
            #                     match submodel_type:
            #                         case "ordinary least squares":        
            #                             sub2model_types=["ordinary least squares random regression"]
            #                         case "decision tree regressor":
            #                             sub2model_types=["decision tree regressor friedman #1"]
            #                         case "decision tree classifier":
            #                             sub2model_types=["decision tree classifier random 2-class classification"]
            #                         case "support vector regressor":
            #                             sub2model_types=["linear support vector regressor"]
            #                         case "nearest neighbors regressor":
            #                             sub2model_types=["k nearest neighbors regressor"]
            #                         case "random forests":
            #                             sub2model_types=["random forest regressor"]
            #                         case "multi-layer perceptrons":
            #                             sub2model_types=["multi-layer perceptron regressor"]
            #                         case "k means":
            #                            sub2model_types=["k means isotropic Gaussian blobs"]
            #                         # case "polynomial regression":
            #                         #     sub2model_types=["quadratic regression"]
            #                     for sub2model_type in sub2model_types:
            #                         sub2model_parameters=submodel_parameters[sub2model_type]
            #                         sub2model_datasets=submodel_datasets[sub2model_type]
            #                         sub2model_pipes=submodel_pipes[sub2model_type]
            #                         sub2model_dataset_plots=Sub2model_Dataset_Plots(sub2model_parameters, sub2model_datasets, sub2model_pipes)
            #                         submodel_dataset_plots[sub2model_type]=sub2model_dataset_plots
            #                     return submodel_dataset_plots
                            
                            
            #                 model_dataset_plots={}
            #                 match model_type:
            #                     case "linear models":        
            #                         submodel_types=["ordinary least squares", "polynomial regression"]
            #                     case "decision trees":
            #                         submodel_types=["decision tree regressor", "decision tree classifier"]
            #                     case "support vector machines":
            #                         submodel_types=["support vector regressor"]
            #                     case "supervised nearest neighbors":
            #                         submodel_types=["nearest neighbors regressor"]
            #                     case "ensembles":
            #                         submodel_types=["random forests"]
            #                     case "neural network models":
            #                         submodel_types=["multi-layer perceptrons"]
            #                     case "clustering":
            #                         submodel_types=["k means"]
            #                 for submodel_type in submodel_types:
            #                     submodel_parameters=model_parameters[submodel_type]
            #                     submodel_datasets=model_datasets[submodel_type]
            #                     submodel_pipes=model_pipes[submodel_type]
            #                     submodel_dataset_plots=Submodel_Dataset_Plots(submodel_parameters, submodel_datasets, submodel_pipes)
            #                     model_dataset_plots[submodel_type]=submodel_dataset_plots
            #                 return model_dataset_plots
                        
            #             learning_dataset_plots={}
            #             match learning_type:
            #                 case "supervised learning":
            #                     model_types=["linear models", "decision trees", "support vector machines", "supervised nearest neighbors", "ensembles", "neural network models"]
            #                 case "unsupervised learning":
            #                     model_types=["clustering"]
            #             for model_type in model_types:
            #                 model_parameters=learning_parameters[model_type]
            #                 model_datasets=learning_datasets[model_type]
            #                 model_pipes=learning_pipes[model_type]
            #                 model_dataset_plots=Model_Dataset_Plots(model_parameters, model_datasets, model_pipes)
            #                 learning_dataset_plots[model_type]=model_dataset_plots
            #             return learning_dataset_plots
                
            #         dataset_plots={}
            #         learning_types=["supervised learning", "unsupervised learning"]
            #         for learning_type in learning_types:
            #             learning_parameters=parameters[learning_type]
            #             learning_datasets=datasets[learning_type]
            #             learning_pipes=pipes[learning_type]
            #             learning_dataset_plots=Learning_Dataset_Plots(learning_parameters, learning_datasets, learning_pipes)
            #             dataset_plots[learning_type]=learning_dataset_plots
            #         return dataset_plots
                
            #     def Score_Vs_DDR_Plots(parameters, scores):
            #         """
            #         Takes the parameters and scores and returns the score vs. DDR plots.
    
            #         Parameters
            #         ----------
            #         parameters : Dict
            #             Parameters of the experiment.
            #         scores : Dict
            #             Scores from the experiment.
    
            #         Returns
            #         -------
            #         score_vs_ddr_plots : Dict
            #             Score vs. DDR plots from the experiment.
    
            #         """    
                
            #         def Learning_Score_Vs_DDR_Plots(learning_parameters, learning_scores):
            #             """
            #             Takes the learning parameters and learning scores and returns the learning score vs. DDR plots.
        
            #             Parameters
            #             ----------
            #             learning_parameters : Dict
            #                 Parameters of the learning type.
            #             learning_scores : Dict
            #                 Scores from the learning type.
        
            #             Returns
            #             -------
            #             learning_score_vs_ddr_plots : Dict
            #                 Score vs. DDR plots from the learning.
        
            #             """
                        
            #             def Model_Score_Vs_DDR_Plots(model_parameters, model_scores):
            #                 """
            #                 Takes the model parameters and model scores and returns the model score vs. DDR plots.
        
            #                 Parameters
            #                 ----------
            #                 model_parameters : Dict
            #                     Parameters of the model type.
            #                 model_scores : Dict
            #                     Scores from the model type.
        
            #                 Returns
            #                 -------
            #                 model_score_vs_ddr_plots : Dict
            #                     Score vs. DDR plots from the model.
        
            #                 """
                            
            #                 def Submodel_Score_Vs_DDR_Plots(submodel_parameters, submodel_scores):
            #                     """
            #                     Takes the submodel parameters and submodel scores and returns the submodel plots.
        
            #                     Parameters
            #                     ----------
            #                     submodel_parameters : Dict
            #                         Parameters of the submodel type.
            #                     submodel_scores : Dict
            #                         Scores from the submodel type.
        
            #                     Returns
            #                     -------
            #                     submodel_score_vs_ddr_plots : Dict
            #                         Score vs. DDR plots from the submodel.
        
            #                     """
                            
            #                     def Sub2model_Score_Vs_DDR_Plots(sub2model_parameters, sub2model_scores):
            #                         """
            #                         Takes the sub2model parameters and sub2model scores and returns the sub2model score vs. DDR plots.
        
            #                         Parameters
            #                         ----------
            #                         sub2model_parameters : Dict
            #                             Parameters of the sub2model type.
            #                         sub2model_scores : Dict
            #                             Scores from the sub2model type.
        
            #                         Returns
            #                         -------
            #                         sub2model_score_vs_ddr_plots : Dict
            #                            Score vs. DDR plots from the sub2model.
        
            #                         """
                                    
            #                         def Sub3model_Score_Vs_DDR_Plots(sub3model_parameters, sub3model_scores):
            #                             """
            #                             Takes the sub3model parameters and sub3model scores and returns the sub3model plots.
        
            #                             Parameters
            #                             ----------
            #                             sub3model_parameters : Dict
            #                                 Parameters of the sub3model type.
            #                             sub3model_scores : Dict
            #                                 Scores from the sub3model type.
        
            #                             Returns
            #                             -------
            #                             sub3model_score_vs_ddr_plots : Dict
            #                                 Score vs. DDR plots from the sub3model.
        
            #                             """
                                        
            #                             sub3model_score_vs_ddr_plots={}
            #                             sub3model_types=[]
            #                             match sub3model_type:
            #                                 case "linear support vector regressor random regression":
            #                                     feature_matrix_ddrs=[]
            #                                     sub4model_parameter_types=sub3model_parameters.keys()
            #                                     feature_matrix_ddrs=[]
            #                                     sub3model_scores_train=[]
            #                                     sub3model_scores_train_err=[]
            #                                     sub3model_scores_test=[]
            #                                     sub3model_scores_test_err=[]
            #                                     for sub4model_parameter_type in sub4model_parameter_types:
            #                                         if sub4model_parameter_type.split()[0]=="matrix":
            #                                             feature_matrix_ddr=float(sub4model_parameter_type.split()[-1])
            #                                             feature_matrix_ddrs.append(feature_matrix_ddr)
            #                                             sub4model_scores=sub3model_scores[sub4model_parameter_type]
            #                                             sub4model_scores_keys=sub4model_scores.keys()
            #                                             sub4model_scores_train=[]
            #                                             sub4model_scores_test=[]
            #                                             for sub4model_scores_key in sub4model_scores_keys:
            #                                                 sub4model_score=sub4model_scores[sub4model_scores_key]
            #                                                 if sub4model_scores_key.split()[-1]=="train":
            #                                                     sub4model_scores_train.append(sub4model_score)    
            #                                                 else:
            #                                                     sub4model_scores_test.append(sub4model_score)
            #                                             sub3model_scores_train.append(sub4model_scores_train)
            #                                             sub3model_scores_test.append(sub4model_scores_test)
            #                                     feature_matrix_ddrs=np.array(feature_matrix_ddrs)
            #                                     sub3model_scores_train=[np.mean(sub4model_scores_train) for sub4model_scores_train in sub3model_scores_train]
            #                                     sub3model_scores_train=np.array(sub3model_scores_train)
            #                                     data_train=np.concatenate((feature_matrix_ddrs.reshape(-1,1), sub3model_scores_train.reshape(-1,1)), axis=1)
            #                                     df_train=pd.DataFrame(data=data_train, columns=["DDR", "NMSE-based Accuracy"])
            #                                     fig_train, ax_train=plt.subplots()
            #                                     sns.scatterplot(data=df_train,x="DDR", y="NMSE-based Accuracy",  ax=ax_train)
            #                                     ax_train.set_title(f"{sub3model_type} NMSE-based training accuracy vs. matrix of features DDR")
            #                                     plt.show(fig_train)
            #                                     #plt.close(fig_train)
            #                                     sub3model_score_vs_ddr_plots["NMSE-based training accuracy vs. matrix of features DDR"]=fig_train
            #                                     sub3model_scores_test=[np.mean(sub4model_scores_test) for sub4model_scores_test in sub3model_scores_test]
            #                                     sub3model_scores_test=np.array(sub3model_scores_test)
            #                                     data_test=np.concatenate((feature_matrix_ddrs.reshape(-1,1), sub3model_scores_test.reshape(-1,1)), axis=1)
            #                                     df_test=pd.DataFrame(data=data_test, columns=["DDR", "NMSE-based Accuracy"])
            #                                     fig_test, ax_test=plt.subplots()
            #                                     sns.scatterplot(data=df_test,x="DDR", y="NMSE-based Accuracy",  ax=ax_test)
            #                                     ax_test.set_title(f"{sub3model_type} NMSE-based training accuracy vs. matrix of features DDR")
            #                                     plt.show(fig_test)
            #                                     #plt.close(fig_test)
            #                                     sub3model_score_vs_ddr_plots["NMSE-based testing accuracy vs. matrix of features DDR"]=fig_test                                            
            #                                 case "k nearest neighbors regressor friedman #1":
            #                                     feature_matrix_ddrs=[]
            #                                     sub4model_parameter_types=sub3model_parameters.keys()
            #                                     feature_matrix_ddrs=[]
            #                                     sub3model_scores_train=[]
            #                                     sub3model_scores_train_err=[]
            #                                     sub3model_scores_test=[]
            #                                     sub3model_scores_test_err=[]
            #                                     for sub4model_parameter_type in sub4model_parameter_types:
            #                                         if sub4model_parameter_type.split()[0]=="matrix":
            #                                             feature_matrix_ddr=float(sub4model_parameter_type.split()[-1])
            #                                             feature_matrix_ddrs.append(feature_matrix_ddr)
            #                                             sub4model_scores=sub3model_scores[sub4model_parameter_type]
            #                                             sub4model_scores_keys=sub4model_scores.keys()
            #                                             sub4model_scores_train=[]
            #                                             sub4model_scores_test=[]
            #                                             for sub4model_scores_key in sub4model_scores_keys:
            #                                                 sub4model_score=sub4model_scores[sub4model_scores_key]
            #                                                 if sub4model_scores_key.split()[-1]=="train":
            #                                                     sub4model_scores_train.append(sub4model_score)    
            #                                                 else:
            #                                                     sub4model_scores_test.append(sub4model_score)
            #                                             sub3model_scores_train.append(sub4model_scores_train)
            #                                             sub3model_scores_test.append(sub4model_scores_test)
            #                                     feature_matrix_ddrs=np.array(feature_matrix_ddrs)
            #                                     sub3model_scores_train=[np.mean(sub4model_scores_train) for sub4model_scores_train in sub3model_scores_train]
            #                                     sub3model_scores_train=np.array(sub3model_scores_train)
            #                                     data_train=np.concatenate((feature_matrix_ddrs.reshape(-1,1), sub3model_scores_train.reshape(-1,1)), axis=1)
            #                                     df_train=pd.DataFrame(data=data_train, columns=["DDR", "NMSE-based Accuracy"])
            #                                     fig_train, ax_train=plt.subplots()
            #                                     sns.scatterplot(data=df_train,x="DDR", y="NMSE-based Accuracy",  ax=ax_train)
            #                                     ax_train.set_title(f"{sub3model_type} NMSE-based training accuracy vs. matrix of features DDR")
            #                                     plt.show(fig_train)
            #                                     #plt.close(fig_train)
            #                                     sub3model_score_vs_ddr_plots["NMSE-based training accuracy vs. matrix of features DDR"]=fig_train
            #                                     sub3model_scores_test=[np.mean(sub4model_scores_test) for sub4model_scores_test in sub3model_scores_test]
            #                                     sub3model_scores_test=np.array(sub3model_scores_test)
            #                                     data_test=np.concatenate((feature_matrix_ddrs.reshape(-1,1), sub3model_scores_test.reshape(-1,1)), axis=1)
            #                                     df_test=pd.DataFrame(data=data_test, columns=["DDR", "NMSE-based Accuracy"])
            #                                     fig_test, ax_test=plt.subplots()
            #                                     sns.scatterplot(data=df_test,x="DDR", y="NMSE-based Accuracy",  ax=ax_test)
            #                                     ax_test.set_title(f"{sub3model_type} NMSE-based training accuracy vs. matrix of features DDR")
            #                                     plt.show(fig_test)
            #                                     #plt.close(fig_test)
            #                                     sub3model_score_vs_ddr_plots["NMSE-based testing accuracy vs. matrix of features DDR"]=fig_test        
            #                                 case "random forest regressor friedman #1":
            #                                     feature_matrix_ddrs=[]
            #                                     sub4model_parameter_types=sub3model_parameters.keys()
            #                                     feature_matrix_ddrs=[]
            #                                     sub3model_scores_train=[]
            #                                     sub3model_scores_train_err=[]
            #                                     sub3model_scores_test=[]
            #                                     sub3model_scores_test_err=[]
            #                                     for sub4model_parameter_type in sub4model_parameter_types:
            #                                         if sub4model_parameter_type.split()[0]=="matrix":
            #                                             feature_matrix_ddr=float(sub4model_parameter_type.split()[-1])
            #                                             feature_matrix_ddrs.append(feature_matrix_ddr)
            #                                             sub4model_scores=sub3model_scores[sub4model_parameter_type]
            #                                             sub4model_scores_keys=sub4model_scores.keys()
            #                                             sub4model_scores_train=[]
            #                                             sub4model_scores_test=[]
            #                                             for sub4model_scores_key in sub4model_scores_keys:
            #                                                 sub4model_score=sub4model_scores[sub4model_scores_key]
            #                                                 if sub4model_scores_key.split()[-1]=="train":
            #                                                     sub4model_scores_train.append(sub4model_score)    
            #                                                 else:
            #                                                     sub4model_scores_test.append(sub4model_score)
            #                                             sub3model_scores_train.append(sub4model_scores_train)
            #                                             sub3model_scores_test.append(sub4model_scores_test)
            #                                     feature_matrix_ddrs=np.array(feature_matrix_ddrs)
            #                                     sub3model_scores_train=[np.mean(sub4model_scores_train) for sub4model_scores_train in sub3model_scores_train]
            #                                     sub3model_scores_train=np.array(sub3model_scores_train)
            #                                     data_train=np.concatenate((feature_matrix_ddrs.reshape(-1,1), sub3model_scores_train.reshape(-1,1)), axis=1)
            #                                     df_train=pd.DataFrame(data=data_train, columns=["DDR", "NMSE-based Accuracy"])
            #                                     fig_train, ax_train=plt.subplots()
            #                                     sns.scatterplot(data=df_train,x="DDR", y="NMSE-based Accuracy",  ax=ax_train)
            #                                     ax_train.set_title(f"{sub3model_type} NMSE-based training accuracy vs. matrix of features DDR")
            #                                     plt.show(fig_train)
            #                                     #plt.close(fig_train)
            #                                     sub3model_score_vs_ddr_plots["NMSE-based training accuracy vs. matrix of features DDR"]=fig_train
            #                                     sub3model_scores_test=[np.mean(sub4model_scores_test) for sub4model_scores_test in sub3model_scores_test]
            #                                     sub3model_scores_test=np.array(sub3model_scores_test)
            #                                     data_test=np.concatenate((feature_matrix_ddrs.reshape(-1,1), sub3model_scores_test.reshape(-1,1)), axis=1)
            #                                     df_test=pd.DataFrame(data=data_test, columns=["DDR", "NMSE-based Accuracy"])
            #                                     fig_test, ax_test=plt.subplots()
            #                                     sns.scatterplot(data=df_test,x="DDR", y="NMSE-based Accuracy",  ax=ax_test)
            #                                     ax_test.set_title(f"{sub3model_type} NMSE-based training accuracy vs. matrix of features DDR")
            #                                     plt.show(fig_test)
            #                                     #plt.close(fig_test)
            #                                     sub3model_score_vs_ddr_plots["NMSE-based testing accuracy vs. matrix of features DDR"]=fig_test        
            #                             return sub3model_score_vs_ddr_plots
                                    
            #                         sub2model_score_vs_ddr_plots={}
            #                         sub3model_types=[]
            #                         feature_matrix_ddrs=[]
            #                         match sub2model_type:
            #                             case "ordinary least squares random regression":
            #                                 feature_matrix_ddrs=[]
            #                                 sub3model_parameter_types=sub2model_parameters.keys()
            #                                 feature_matrix_ddrs=[]
            #                                 sub2model_scores_train=[]
            #                                 sub2model_scores_train_err=[]
            #                                 sub2model_scores_test=[]
            #                                 sub2model_scores_test_err=[]
            #                                 for sub3model_parameter_type in sub3model_parameter_types:
            #                                     if sub3model_parameter_type.split()[0]=="matrix":
            #                                         feature_matrix_ddr=float(sub3model_parameter_type.split()[-1])
            #                                         feature_matrix_ddrs.append(feature_matrix_ddr)
            #                                         sub3model_scores=sub2model_scores[sub3model_parameter_type]
            #                                         sub3model_scores_keys=sub3model_scores.keys()
            #                                         sub3model_scores_train=[]
            #                                         sub3model_scores_test=[]
            #                                         for sub3model_scores_key in sub3model_scores_keys:
            #                                             sub3model_score=sub3model_scores[sub3model_scores_key]
            #                                             if sub3model_scores_key.split()[-1]=="train":
            #                                                 sub3model_scores_train.append(sub3model_score)    
            #                                             else:
            #                                                 sub3model_scores_test.append(sub3model_score)
            #                                         sub2model_scores_train.append(sub3model_scores_train)
            #                                         sub2model_scores_test.append(sub3model_scores_test)
            #                                 feature_matrix_ddrs=np.array(feature_matrix_ddrs)
            #                                 sub2model_scores_train=[np.mean(sub3model_scores_train) for sub3model_scores_train in sub2model_scores_train]
            #                                 sub2model_scores_train=np.array(sub2model_scores_train)
            #                                 data_train=np.concatenate((feature_matrix_ddrs.reshape(-1,1), sub2model_scores_train.reshape(-1,1)), axis=1)
            #                                 df_train=pd.DataFrame(data=data_train, columns=["DDR", "NMSE-based Accuracy"])
            #                                 fig_train, ax_train=plt.subplots()
            #                                 sns.scatterplot(data=df_train,x="DDR", y="NMSE-based Accuracy",  ax=ax_train)
            #                                 ax_train.set_title(f"{sub2model_type} NMSE-based training accuracy vs. matrix of features DDR")
            #                                 plt.show(fig_train)
            #                                 #plt.close(fig_train)
            #                                 sub2model_score_vs_ddr_plots["NMSE-based training accuracy vs. matrix of features DDR"]=fig_train
            #                                 sub2model_scores_test=[np.mean(sub3model_scores_test) for sub3model_scores_test in sub2model_scores_test]
            #                                 sub2model_scores_test=np.array(sub2model_scores_test)
            #                                 data_test=np.concatenate((feature_matrix_ddrs.reshape(-1,1), sub2model_scores_test.reshape(-1,1)), axis=1)
            #                                 df_test=pd.DataFrame(data=data_test, columns=["DDR", "NMSE-based Accuracy"])
            #                                 fig_test, ax_test=plt.subplots()
            #                                 sns.scatterplot(data=df_test,x="DDR", y="NMSE-based Accuracy",  ax=ax_test)
            #                                 ax_test.set_title(f"{sub2model_type} NMSE-based training accuracy vs. matrix of features DDR")
            #                                 plt.show(fig_test)
            #                                 #plt.close(fig_test)
            #                                 sub2model_score_vs_ddr_plots["NMSE-based testing accuracy vs. matrix of features DDR"]=fig_test   
            #                             case "decision tree regressor friedman #1":
            #                                 feature_matrix_ddrs=[]
            #                                 sub3model_parameter_types=sub2model_parameters.keys()
            #                                 feature_matrix_ddrs=[]
            #                                 sub2model_scores_train=[]
            #                                 sub2model_scores_train_err=[]
            #                                 sub2model_scores_test=[]
            #                                 sub2model_scores_test_err=[]
            #                                 for sub3model_parameter_type in sub3model_parameter_types:
            #                                     if sub3model_parameter_type.split()[0]=="matrix":
            #                                         feature_matrix_ddr=float(sub3model_parameter_type.split()[-1])
            #                                         feature_matrix_ddrs.append(feature_matrix_ddr)
            #                                         sub3model_scores=sub2model_scores[sub3model_parameter_type]
            #                                         sub3model_scores_keys=sub3model_scores.keys()
            #                                         sub3model_scores_train=[]
            #                                         sub3model_scores_test=[]
            #                                         for sub3model_scores_key in sub3model_scores_keys:
            #                                             sub3model_score=sub3model_scores[sub3model_scores_key]
            #                                             if sub3model_scores_key.split()[-1]=="train":
            #                                                 sub3model_scores_train.append(sub3model_score)    
            #                                             else:
            #                                                 sub3model_scores_test.append(sub3model_score)
            #                                         sub2model_scores_train.append(sub3model_scores_train)
            #                                         sub2model_scores_test.append(sub3model_scores_test)
            #                                 feature_matrix_ddrs=np.array(feature_matrix_ddrs)
            #                                 sub2model_scores_train=[np.mean(sub3model_scores_train) for sub3model_scores_train in sub2model_scores_train]
            #                                 sub2model_scores_train=np.array(sub2model_scores_train)
            #                                 data_train=np.concatenate((feature_matrix_ddrs.reshape(-1,1), sub2model_scores_train.reshape(-1,1)), axis=1)
            #                                 df_train=pd.DataFrame(data=data_train, columns=["DDR", "NMSE-based Accuracy"])
            #                                 fig_train, ax_train=plt.subplots()
            #                                 sns.scatterplot(data=df_train,x="DDR", y="NMSE-based Accuracy",  ax=ax_train)
            #                                 ax_train.set_title(f"{sub2model_type} NMSE-based training accuracy vs. matrix of features DDR")
            #                                 plt.show(fig_train)
            #                                 #plt.close(fig_train)
            #                                 sub2model_score_vs_ddr_plots["NMSE-based training accuracy vs. matrix of features DDR"]=fig_train
            #                                 sub2model_scores_test=[np.mean(sub3model_scores_test) for sub3model_scores_test in sub2model_scores_test]
            #                                 sub2model_scores_test=np.array(sub2model_scores_test)
            #                                 data_test=np.concatenate((feature_matrix_ddrs.reshape(-1,1), sub2model_scores_test.reshape(-1,1)), axis=1)
            #                                 df_test=pd.DataFrame(data=data_test, columns=["DDR", "NMSE-based Accuracy"])
            #                                 fig_test, ax_test=plt.subplots()
            #                                 sns.scatterplot(data=df_test,x="DDR", y="NMSE-based Accuracy",  ax=ax_test)
            #                                 ax_test.set_title(f"{sub2model_type} NMSE-based training accuracy vs. matrix of features DDR")
            #                                 plt.show(fig_test)
            #                                 #plt.close(fig_test)
            #                                 sub2model_score_vs_ddr_plots["NMSE-based testing accuracy vs. matrix of features DDR"]=fig_test  
            #                             case "decision tree classifier random 2-class classification":
            #                                 feature_matrix_ddrs=[]
            #                                 sub3model_parameter_types=sub2model_parameters.keys()
            #                                 feature_matrix_ddrs=[]
            #                                 sub2model_scores_train=[]
            #                                 sub2model_scores_train_err=[]
            #                                 sub2model_scores_test=[]
            #                                 sub2model_scores_test_err=[]
            #                                 for sub3model_parameter_type in sub3model_parameter_types:
            #                                     if sub3model_parameter_type.split()[0]=="matrix":
            #                                         feature_matrix_ddr=float(sub3model_parameter_type.split()[-1])
            #                                         feature_matrix_ddrs.append(feature_matrix_ddr)
            #                                         sub3model_scores=sub2model_scores[sub3model_parameter_type]
            #                                         sub3model_scores_keys=sub3model_scores.keys()
            #                                         sub3model_scores_train=[]
            #                                         sub3model_scores_test=[]
            #                                         for sub3model_scores_key in sub3model_scores_keys:
            #                                             sub3model_score=sub3model_scores[sub3model_scores_key]
            #                                             if sub3model_scores_key.split()[-1]=="train":
            #                                                 sub3model_scores_train.append(sub3model_score)    
            #                                             else:
            #                                                 sub3model_scores_test.append(sub3model_score)
            #                                         sub2model_scores_train.append(sub3model_scores_train)
            #                                         sub2model_scores_test.append(sub3model_scores_test)
            #                                 feature_matrix_ddrs=np.array(feature_matrix_ddrs)
            #                                 sub2model_scores_train=[np.mean(sub3model_scores_train) for sub3model_scores_train in sub2model_scores_train]
            #                                 sub2model_scores_train=np.array(sub2model_scores_train)
            #                                 data_train=np.concatenate((feature_matrix_ddrs.reshape(-1,1), sub2model_scores_train.reshape(-1,1)), axis=1)
            #                                 df_train=pd.DataFrame(data=data_train, columns=["DDR", "F1 Score"] )
            #                                 fig_train, ax_train=plt.subplots()
            #                                 sns.scatterplot(data=df_train,x="DDR", y="F1 Score",  ax=ax_train)
            #                                 ax_train.set_title(f"{sub2model_type} F1 Score vs. matrix of features DDR")
            #                                 plt.show(fig_train)
            #                                 #plt.close(fig_train)
            #                                 sub2model_score_vs_ddr_plots["F1 Score vs. matrix of features DDR"]=fig_train
            #                                 sub2model_scores_test=[np.mean(sub3model_scores_test) for sub3model_scores_test in sub2model_scores_test]
            #                                 sub2model_scores_test=np.array(sub2model_scores_test)
            #                                 data_test=np.concatenate((feature_matrix_ddrs.reshape(-1,1), sub2model_scores_test.reshape(-1,1)), axis=1)
            #                                 df_test=pd.DataFrame(data=data_test, columns=["DDR", "F1 Score"])
            #                                 fig_test, ax_test=plt.subplots()
            #                                 sns.scatterplot(data=df_test,x="DDR", y="F1 Score",  ax=ax_test)
            #                                 ax_test.set_title(f"{sub2model_type} F1 Score vs. matrix of features DDR")
            #                                 plt.show(fig_test)
            #                                 #plt.close(fig_test)
            #                                 sub2model_score_vs_ddr_plots["F1 Score vs. matrix of features DDR"]=fig_test  
            #                             case "linear support vector regressor":
            #                                 sub3model_types=["linear support vector regressor random regression"]
            #                             case "k nearest neighbors regressor":
            #                                 sub3model_types=["k nearest neighbors regressor friedman #1"]
            #                             case "random forest regressor":
            #                                 sub3model_types=["random forest regressor friedman #1"]
            #                             case "multi-layer perceptron regressor":
            #                                 sub3model_types=["multi-layer perceptron regressor friedman #1"]
            #                             case "k means isotropic Gaussian blobs":
            #                                 feature_matrix_ddrs=[]
            #                                 sub3model_parameter_types=sub2model_parameters.keys()
            #                                 feature_matrix_ddrs=[]
            #                                 sub2model_scores_all=[]
            #                                 sub2model_scores_all_err=[]
            #                                 for sub3model_parameter_type in sub3model_parameter_types:
            #                                     if sub3model_parameter_type.split()[0]=="matrix":
            #                                         feature_matrix_ddr=float(sub3model_parameter_type.split()[-1])
            #                                         feature_matrix_ddrs.append(feature_matrix_ddr)
            #                                         sub3model_scores=sub2model_scores[sub3model_parameter_type]
            #                                         sub3model_scores_keys=sub3model_scores.keys()
            #                                         sub3model_scores_all=[]
            #                                         for sub3model_scores_key in sub3model_scores_keys:
            #                                             sub3model_score=sub3model_scores[sub3model_scores_key]
            #                                             sub3model_scores_all.append(sub3model_score)    
            #                                         sub2model_scores_all.append(sub3model_scores_all)
            #                                 feature_matrix_ddrs=np.array(feature_matrix_ddrs)
            #                                 sub2model_scores_all=[np.mean(sub3model_scores) for sub3model_scores in sub2model_scores_all]
            #                                 sub2model_scores_all=np.array(sub2model_scores_all)
            #                                 data=np.concatenate((feature_matrix_ddrs.reshape(-1,1), sub2model_scores_all.reshape(-1,1)), axis=1)
            #                                 df=pd.DataFrame(data=data, columns=["DDR", "V-Measure"])
            #                                 fig, ax=plt.subplots()
            #                                 sns.scatterplot(data=df,x="DDR", y="V-Measure",  ax=ax)
            #                                 ax.set_title(f"{sub2model_type} V-Measure vs. matrix of features DDR")
            #                                 plt.show(fig)
            #                                 #plt.close(fig)
            #                                 sub2model_score_vs_ddr_plots["V-Measure vs. matrix of features DDR"]=fig
            #                             # case "quadratic regression":
            #                             #     sub3model_types=["quadratic regression random quadratic regression"]
            #                         for sub3model_type in sub3model_types:
            #                             sub3model_parameters=sub2model_parameters[sub3model_type]
            #                             sub3model_scores=sub2model_scores[sub3model_type]
            #                             sub3model_score_vs_ddr_plots=Sub3model_Score_Vs_DDR_Plots(sub3model_parameters, sub3model_scores)
            #                             sub2model_score_vs_ddr_plots[sub3model_type]=sub3model_score_vs_ddr_plots
            #                         return sub2model_score_vs_ddr_plots
                            
            #                     submodel_score_vs_ddr_plots={}
            #                     sub2model_types=[]
            #                     match submodel_type:
            #                         case "ordinary least squares":        
            #                             sub2model_types=["ordinary least squares random regression"]
            #                         case "decision tree regressor":
            #                             sub2model_types=["decision tree regressor friedman #1"]
            #                         case "decision tree classifier":
            #                             sub2model_types=["decision tree classifier random 2-class classification"]
            #                         case "support vector regressor":
            #                             sub2model_types=["linear support vector regressor"]
            #                         case "nearest neighbors regressor":
            #                             sub2model_types=["k nearest neighbors regressor"]
            #                         case "random forests":
            #                             sub2model_types=["random forest regressor"]
            #                         case "multi-layer perceptrons":
            #                             sub2model_types=["multi-layer perceptron regressor"]
            #                         case "k means":
            #                             sub2model_types=["k means isotropic Gaussian blobs"]
            #                         # case "polynomial regression":
            #                         #     sub2model_types=["quadratic regression"]
            #                     for sub2model_type in sub2model_types:
            #                         sub2model_parameters=submodel_parameters[sub2model_type]
            #                         sub2model_scores=submodel_scores[sub2model_type]
            #                         sub2model_score_vs_ddr_plots=Sub2model_Score_Vs_DDR_Plots(sub2model_parameters, sub2model_scores)
            #                         submodel_score_vs_ddr_plots[sub2model_type]=sub2model_score_vs_ddr_plots
            #                     return submodel_score_vs_ddr_plots
                            
                            
            #                 model_score_vs_ddr_plots={}
            #                 match model_type:
            #                     case "linear models":        
            #                         submodel_types=["ordinary least squares", "polynomial regression"]
            #                     case "decision trees":
            #                         submodel_types=["decision tree regressor", "decision tree classifier"]
            #                     case "support vector machines":
            #                         submodel_types=["support vector regressor"]
            #                     case "supervised nearest neighbors":
            #                         submodel_types=["nearest neighbors regressor"]
            #                     case "ensembles":
            #                         submodel_types=["random forests"]
            #                     case "neural network models":
            #                         submodel_types=["multi-layer perceptrons"]
            #                     case "clustering":
            #                         submodel_types=["k means"]
            #                 for submodel_type in submodel_types:
            #                     submodel_parameters=model_parameters[submodel_type]
            #                     submodel_scores=model_scores[submodel_type]
            #                     submodel_score_vs_ddr_plots=Submodel_Score_Vs_DDR_Plots(submodel_parameters, submodel_scores)
            #                     model_score_vs_ddr_plots[submodel_type]=submodel_score_vs_ddr_plots
            #                 return model_score_vs_ddr_plots
                        
            #             learning_score_vs_ddr_plots={}
            #             match learning_type:
            #                 case "supervised learning":
            #                     model_types=["linear models", "decision trees", "support vector machines", "supervised nearest neighbors", "ensembles", "neural network models"]
            #                 case "unsupervised learning":
            #                     model_types=["clustering"]
            #             for model_type in model_types:
            #                 model_parameters=learning_parameters[model_type]
            #                 model_scores=learning_scores[model_type]
            #                 model_score_vs_ddr_plots=Model_Score_Vs_DDR_Plots(model_parameters, model_scores)
            #                 learning_score_vs_ddr_plots[model_type]=model_score_vs_ddr_plots
            #             return learning_score_vs_ddr_plots
                
            #         score_vs_ddr_plots={}
            #         learning_types=["supervised learning", "unsupervised learning"]
            #         for learning_type in learning_types:
            #             learning_parameters=parameters[learning_type]
            #             learning_scores=scores[learning_type]
            #             learning_score_vs_ddr_plots=Learning_Score_Vs_DDR_Plots(learning_parameters, learning_scores)
            #             score_vs_ddr_plots[learning_type]=learning_score_vs_ddr_plots
            #         return score_vs_ddr_plots
            
            #     plots={}
            #     dataset_plots=Dataset_Plots(parameters, datasets, pipes)
            #     plots["dataset plots"]=dataset_plots
            #     score_vs_ddr_plots=Score_Vs_DDR_Plots(parameters, scores)
            #     plots["score vs. DDR plots"]=score_vs_ddr_plots
            #     return plots
                
            outputs={}
            parameters=inputs["parameters"]
            datasets=inputs["datasets"]
            pipes=inputs["pipes"]
            scores=Scores(datasets, pipes)
            table=Table(scores)
            # plots=Plots(parameters, datasets, pipes, scores)
            outputs["parameters"]=parameters
            outputs["datasets"]=datasets
            outputs["pipes"]=pipes
            outputs["scores"]=scores
            outputs["table"]=table
            # outputs["plots"]=plots
            return outputs
        
        # def Tables(outputs):
        #     tables={}
        #     parameters=outputs["parameters"] 
        #     datasets=outputs["datasets"]
        #     pipes=outputs["pipes"]
        #     scores=outputs["scores"]
        #     plots=outputs["plots"]
        #     parameter_tables=Parameter_Tables(parameters)
        #     dataset_tables=Dataset_Tables(datasets)
        #     pipe_tables=Pipe_Tables(pipes)
        #     score_tables=Score_Tables(scores)
        #     plot_tables=Plot_Tables
                
        #     return tables
        
        experiment={}
        inputs=Inputs(parameters)
        outputs=Outputs(inputs)    
        #tables=Tables(outputs)
        experiment["parameters"]=parameters
        experiment["inputs"]=inputs
        experiment["outputs"]=outputs
        #experiment["tables"]=tables
        return experiment    
    
    
    model_hyperparameters_models=[]
    model_hyperparameters_hyperparameters=[]
    model_hyperparameters_hyperparameter_values=[]
    
    def Save_Experiment(experiment):
        """
        Saves the experiment to a local file.
    
        Parameters
        ----------
        experiment : Dict
            The experiment and its conditions.
    
        Returns
        -------
        None.
    
        """
    
        # def Save_Subexperiment(subexperiment, subexperiment_type):
        #     """
        #     Saves the subexperiment to a local file.
    
        #     Parameters
        #     ----------
        #     subexperiment : Dict
        #         The subexperiment and its conditions.
    
        #     Returns
        #     -------
        #     None.
    
        #     """
            
        #     def Save_Sub2experiment(sub2experiment, sub2experiment_type):
        #         """
        #         Saves the sub2experiment to a local file.
    
        #         Parameters
        #         ----------
        #         sub2experiment : Dict
        #             The sub2experiment and its conditions.
    
        #         Returns
        #         -------
        #         None.
    
        #         """
            
        #         def Save_Sub3experiment(sub3experiment, sub3experiment_type):
        #             """
        #             Saves the sub3experiment to a local file.
    
        #             Parameters
        #             ----------
        #             sub3experiment : Dict
        #                 The sub3experiment and its conditions.
    
        #             Returns
        #             -------
        #             None.
    
        #             """
            
        #             def Save_Sub4experiment(sub4experiment, sub4experiment_type):
        #                 """
        #                 Saves the sub4experiment to a local file.
    
        #                 Parameters
        #                 ----------
        #                 sub4experiment : Dict
        #                     The sub4experiment and its conditions.
    
        #                 Returns
        #                 -------
        #                 None.
    
        #                 """
                        
                        
        #                 match sub4experiment_type:
        #                     case "decision tree classifier":
        #                         sub5experiment_types=["decision tree classifier random 2-class classification", "maximum depth", "minimum number of samples required to be at a leaf", "random state"]
        #                         for sub5experiment_type in sub5experiment_types:
        #                             if sub5experiment_type!="decision tree classifier random 2-class classification":
        #                                 model_hyperparameters_models.append(sub4experiment_type)
        #                                 model_hyperparameters_hyperparameters.append(sub5experiment_type)
        #                                 model_hyperparameters_hyperparameter_values.append(sub4experiment[sub5experiment_type])
        #                             else:
        #                                 sub5experiment=sub4experiment[sub5experiment_type]
        #                                 Save_Sub24experiment(sub5experiment, sub5experiment_type)
        #                 # for sub5experiment_type in sub5experiment_types:
        #                 #     sub5experiment=sub4experiment[sub5experiment_type]
        #                 #     Save_Sub24experiment(sub5experiment, sub5experiment_type)
            
        #             match sub3experiment_type:
        #                 case "decision trees":
        #                     sub4experiment_types=["decision tree classifier", "decision tree regressor"]
                        
        #             for sub4experiment_type in sub4experiment_types:
        #                 sub4experiment=sub3experiment[sub4experiment_type]
        #                 Save_Sub4experiment(sub4experiment, sub4experiment_type)
            
        #         match sub2experiment_type:
        #             case "supervised learning":
        #                 sub3experiment_types=["decision trees", "ensembles", "linear models", "neural network models", "supervised nearest neighbors", "support vector machines"] 
                
        #         for sub3experiment_type in sub3experiment_types:
        #             sub3experiment=sub2experiment[sub3experiment_type]
        #             Save_Sub3experiment(sub3experiment, sub3experiment_type)
            
        #     match subexperiment_type:
        #         case "parameters":
        #             sub2experiment_types=["supervised learning", "unsupervised learning"]
                    
            
        #     for sub2experiment_type in sub2experiment_types:
        #         sub2experiment=subexperiment[sub2experiment_type]
        #         Save_Sub2experiment(sub2experiment, sub2experiment_type)
    
        outputs=experiment["outputs"]
        table=outputs["table"]
        table.to_csv("Scores_Feature_Matrix_DDRs_Model_Types.csv")
    
        
    
    parameters=Parameters()
    experiment=Experiment(parameters)
    Save_Experiment(experiment)
