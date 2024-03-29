import numpy as np


class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.2,
            (1, 1): 0.5
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.4,
            (1, 1): 0.3
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.3,
            (1, 1): 0.4
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.1,
            (0, 0, 1): 0.05,
            (0, 1, 0): 0.1,
            (0, 1, 1): 0.05,
            (1, 0, 0): 0.1,
            (1, 0, 1): 0.05,
            (1, 1, 0): 0.3,
            (1, 1, 1): 0.25,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are dependent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ###########################################################################
        ###########################################################################
        for x in [0, 1]:
            for y in [0, 1]:
                if not np.isclose(X_Y[(x, y)], X[x] * Y[y]):  # if P(X∩Y) ≠ P(X)*P(Y)
                    return True # Dependent

        # if P(X∩Y) = P(X)*P(Y) for every value in X, Y - They are independent
        return False
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are independent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for x in [0, 1]:
            for y in [0, 1]:
                for c in [0, 1]:
                    if not np.isclose(X_Y_C[(x, y, c)], X_C[x, c] * Y_C[y, c]):  # if P(X∩Y) ≠ P(X)*P(B)
                        return True  # independent

            # dependent
        return False
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    ###########################################################################
    ###########################################################################
    log_p = (rate**k) * (np.e**(np.negative(rate)))
    log_p /= np.math.factorial(k)
    log_p = np.log(log_p)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return log_p


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = None
    ###########################################################################
    ###########################################################################
    likelihoods = [np.sum([poisson_log_pmf(s, r) for s in samples]) for r in rates]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    ###########################################################################
    ###########################################################################

    rate = rates[np.argmax(likelihoods)]  # position of the maximum likelihood

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return rate


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    ###########################################################################
    ###########################################################################
    mean = np.average(samples)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean


def normal_pdf(x, mean, std):
    """
    Calculate normal density function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    ###########################################################################
    ###########################################################################
    p = 1 / np.sqrt(2 * np.pi * (std ** 2))
    calc_exponent = np.negative(((x - mean) ** 2) / (2 * (std ** 2)))

    p *= np.e ** calc_exponent
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p


class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        ###########################################################################
        self.dataset = dataset
        self.class_value = class_value

        self.class_label = dataset[:, -1]
        self.class_data = dataset[self.class_label == class_value, :-1]

        self.mean = np.mean(self.class_label, axis=0)
        self.std = np.std(self.class_label, axis=0)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        ###########################################################################
        prior = len(self.class_data) / len(self.dataset)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        ###########################################################################

        normal_val = normal_pdf(x, self.mean, self.std)
        likelihood = np.prod(normal_val)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior probability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        ###########################################################################
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior


class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        ###########################################################################
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        ###########################################################################
        if self.ccd0.get_instance_posterior(x) < self.ccd1.get_instance_posterior(x):
            pred = 1
        else:
            pred = 0
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    ###########################################################################
    ###########################################################################

    # Predict classes using the MAP classifier
    predicted_labels = np.array([map_classifier.predict(instance) for instance in test_set])

    # Calculate the accuracy
    true_labels = test_set[:, -1]
    acc = np.mean(predicted_labels == true_labels)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return acc


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.

    Returns the normal distribution pdf according to the given mean and var for the given x.
    """
    pdf = None
    ###########################################################################
    ###########################################################################
    x = np.delete(x, -1)

    pdf = (2 * np.pi) ** (-len(x) / 2) * (np.linalg.det(cov) ** -0.5)
    pdf *= np.e ** (-0.5 * (x - mean).T.dot(np.linalg.inv(cov).dot(x - mean)))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf


class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.

        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        ###########################################################################
        self.dataset = dataset
        self.class_value = class_value

        self.class_label = self.dataset[:, -1]
        self.class_data = dataset[self.class_label == class_value, :-1]

        self.mean = np.mean(self.class_data, axis=0)
        self.std = np.std(self.class_data, axis=0)
        self.cov = np.cov(self.class_data.T[0], self.class_data.T[1])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        ###########################################################################
        prior = len(self.class_data) / len(self.dataset)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = 1
        ###########################################################################
        ###########################################################################
        for i in range(len(x) - 1):
            likelihood *= multi_normal_pdf(x, self.mean, self.cov)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        ###########################################################################
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior


class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        ###########################################################################
        if self.ccd0.get_prior() < self.ccd1.get_prior():
            pred = 1
        else:
            pred = 0
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred


class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        ###########################################################################
        if self.ccd0.get_instance_likelihood(x) < self.ccd1.get_instance_likelihood(x):
            pred = 1
        else:
            pred = 0
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred


EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        ###########################################################################
        ###########################################################################
        self.dataset = dataset
        self.class_value = class_value

        self.class_label = dataset[:, -1]
        self.class_data = dataset[self.class_label == class_value, :-1]

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        ###########################################################################

        prior = len(self.class_data) / len(self.dataset)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = 1  # initialized to 1 for the product calculations
        ###########################################################################
        ###########################################################################
        for j, attribute in enumerate(x):
            n_i_j = np.sum(self.class_data[:, j] == attribute)  # count instances where attribute matches the given

            if n_i_j == 0:  # if the attribute doesn't exist in the training data
                likelihood *= EPSILLON
                continue

            n_i = len(self.class_data)  # num of training instances in the class
            v_j = len(np.unique(self.dataset[:, j]))  # number of possible values

            likelihood *= (n_i_j + 1) / (n_i + v_j)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        ###########################################################################
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        ###########################################################################
        if self.ccd0.get_instance_posterior(x) < self.ccd1.get_instance_posterior(x):
            pred = 1
        else:
            pred = 0
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        ###########################################################################
        ###########################################################################
        features = test_set[:, :-1]
        true_labels = test_set[:, -1]

        # Predict classes using the MAP classifier
        predicted_labels = np.array([self.predict(instance) for instance in features])

        # Calculate the accuracy
        acc = np.mean(predicted_labels == true_labels)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return acc


