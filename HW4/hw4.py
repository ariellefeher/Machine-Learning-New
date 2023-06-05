import numpy as np


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    """
     Helper function to fit that calculates the hypothesis using the sigmoid
    """
    def compute_hypothesis(self, X):
        hypothesis = 1 + np.math.e ** (-np.dot(X, self.theta.T))
        hypothesis = 1 / hypothesis

        return hypothesis

    """
     Helper function to fit that calculates the cost function
    """
    def compute_cost(self, X, y, hypothesis):

        J = (-1 / len(X)) * np.sum((y * np.log(hypothesis)) + ((1 - y) * (np.log(1 - hypothesis))) )
        return J

    """
     Helper function to fit that calculates the gradient descent efficiently
     Based on my implementation from HW1
    """
    def gradient_descent(self, X, y):
       #m = X.shape[0]  # number of instances

        m = y.size  # number of labels

        for i in range(self.n_iter):

            hypothesis = self.compute_hypothesis(X)

            cost_value = self.compute_cost(X, y, hypothesis)
            self.Js.append(cost_value)

            if i > 0 and (self.Js[i - 1] - self.Js[i]) < self.eps:
                break

           # self.theta = self.theta - (self.eta / m) * np.dot(X.T, hypothesis)

            self.theta = self.theta - ( self.eta * (np.dot(X.T, hypothesis - y) / m ))
            self.thetas.append(self.theta)

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        ###########################################################################
        self.theta = np.random.random(size=3)  # initialize theta

        # apply bias trick (implementation from HW1)
        X = np.column_stack((np.ones(X.shape[0]), X))

        self.gradient_descent(X, y)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        ###########################################################################
        ###########################################################################

        # apply bias trick (implementation from HW1)
        X = np.column_stack((np.ones(X.shape[0]), X))

        for i in range(X.shape[0]):
             preds.append(1) if self.compute_hypothesis(X[i]) > 0.5 else preds.append(0)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """
    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    ###########################################################################

    # for storing the accuracy calculated per fold
    accuracies = []

    # Step 1: splitting the data into folds
    indices = np.random.permutation(X.shape[0])  # shuffle the indices of data
    fold_indices = np.array_split(indices, folds)

    # Step 2: Training and Testing the data for each division of the folds
    for i in range(folds):
        # Step 2.1: dividing the data into training and testing based on the folds

        train_idx = np.concatenate(fold_indices[:i] + fold_indices[i+1:])
        test_idx = fold_indices[i]

        # create training and test sets
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Step 2.2: Training the training data and predicting accordingly

        algo.fit(X_train, y_train)
        y_predict = algo.predict(X_test)

        # calculate accuracy
        accuracy = np.mean(y_predict == y_test)
        accuracies.append(accuracy)

    # Step 3: pick the best accuracy
    cv_accuracy = np.max(accuracies)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal density function for a given data,
    mean and standard deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    ###########################################################################

    frac = 1 / (sigma * np.sqrt(2 * np.pi))
    calc_exponent = np.exp(-0.5 * (((data - mu) / sigma) ** 2))

    p = frac * calc_exponent

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

# TODO: Fix Class
class EM(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM process
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = []
        self.weights = []
        self.mus = []
        self.sigmas = []
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        ###########################################################################
        # # self.responsibilities = np.zeros((data.shape[0], self.k))
        # self.weights = np.ones(self.k) / self.k
        #
        # # random initialization strategy: divide the data into k partitions
        # # calculate the mus and the sigmas based on each partition
        #
        # data_split = np.array_split(data, self.k)
        #
        # self.mus = np.array([np.mean(partition) for partition in data_split])
        # self.sigmas = np.array([np.std(partition) for partition in data_split])

        temp_data = np.split(data, self.k)

        for i in range(self.k):
            self.weights = np.append(self.weights, 1 / self.k)
            self.mus = np.append(self.mus, [np.mean(temp_data[i])])
            self.sigmas = np.append(self.sigmas, [np.std(temp_data[i])])

            ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        ###########################################################################
        # normalized_responses = np.zeros(shape = (len(data), self.k))
        #
        # for j in range(self.k):
        #
        #     for i in range(len(data)):
        #         normal_vals = norm_pdf(data[i], self.mus[j], self.sigmas[j])
        #         normalized_responses[i][j] = self.weights[j] * normal_vals
        #
        # for i in range(data.shape[0]):
        #
        #     for j in range(self.k):
        #         normalized_responses[i][j] /= np.sum(normalized_responses[i])
        #
        # return normalized_responses

        respons = np.zeros(shape=(len(data), self.k))

        # calculate for each instance the probabilty that it came from gaussian k
        for j in range(self.k):
            for i in range(len(data)):
                respons[i][j] = self.weights[j] * norm_pdf(data[i], self.mus[j], self.sigmas[j])

        # the responsibilities
        for i in range(len(data)):
            sumRes = np.sum(respons[i])
            for j in range(self.k):
                respons[i][j] /= sumRes

        return respons

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        ###########################################################################
        # save the shapes of the data
        rows = self.responsibilities.shape[0]
        columns = self.responsibilities.shape[1]

        # initialize arrays in the right size with zeros
        mus_new = np.zeros(columns)
        sigmas_new = np.zeros(columns)

        # calc the full weights array
        ws_new = self.responsibilities.sum(axis=0) / rows

        # run on all the ranks array and calculate mu and sigma array
        for i in range(columns):
            mus_new[i] = (1 / (ws_new[i] * rows)) * ((data[:] * self.responsibilities[:, i]).sum())
            sigmas_new[i] = np.sqrt(
                (1 / (ws_new[i] * rows)) * (self.responsibilities[:, i] * np.square(data[:] - mus_new[i])).sum())

        return ws_new, mus_new, sigmas_new

        # num_columns = self.responsibilities.shape[1]
        # num_rows = self.responsibilities.shape[0]
        #
        # weights_new = self.responsibilities.sum(axis=0) / num_rows
        #
        # weights_matrix = (1 / (weights_new * num_rows))
        #
        # new_mus = weights_matrix * np.dot(self.responsibilities.T, data)
        #
        # data_diff = data - new_mus[:, np.newaxis]
        # data_diff_squared = np.square(data_diff)
        #
        # weighted_data_diff = self.responsibilities * data_diff_squared
        # new_sigmas = np.sqrt(weights_matrix * np.sum(weighted_data_diff, axis=0))
        #
        # return weights_new, new_mus, new_sigmas

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def cost_function(self, data, i):
        return np.sum(-1 * np.log(self.weights[i] * norm_pdf(data, self.mus[i], self.sigmas[i])))

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization functions to estimate the distribution parameters.
        Store the parameters in the attributes of the EM object.
        Stop the function when the difference between the previous cost and the current cost is less than the specified epsilon
        or when the maximum number of iterations is reached.

        Parameters:
        - data: The input data for training the model.
        """
        guessin_current_cost = []
        guessin_prev_cost = np.zeros(self.k)
        difference = np.infty
        iterations = 0

        self.init_params(data)

        while iterations < self.n_iter and difference > self.eps:
            self.responsibilities = self.expectation(data)
            self.weights, self.mus, self.sigmas = self.maximization(data)
            for i in range(self.k):
                cost = self.cost_function(data, i)
                guessin_current_cost.append(cost)

            difference = np.max([abs(np.array(cost) - np.array(guessin_prev_cost))])
            iterations += 1
            guessin_prev_cost = cost
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    ###########################################################################
    normal_vals = norm_pdf(data, mus, sigmas)
    pdf = np.sum(weights * normal_vals)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

# TODO: Fix Class
class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.classes = [0, 1]
        self.priors = {}
        self.gmm_params = {}

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        ###########################################################################
        for class_value in self.classes:
            class_data = X[y == class_value]
            self.priors[class_value] = len(class_data) / len(X)

            em_X1 = EM(k=self.k)
            em_X2 = EM(k=self.k)

            em_X1.fit(class_data[:, 0])
            em_X2.fit(class_data[:, 1])

            self.gmm_params[class_value] = (em_X1, em_X2)

    ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    def _calculate_likelihood(self, x, em_X1, em_X2):
        fX1 = np.sum([em_X1.weights[i] * norm_pdf(x[0], em_X1.mus[i], em_X1.sigmas[i]) for i in range(self.k)])
        fX2 = np.sum([em_X2.weights[i] * norm_pdf(x[1], em_X2.mus[i], em_X2.sigmas[i]) for i in range(self.k)])
        return fX1 * fX2

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        # preds = None
        ###########################################################################
        ###########################################################################
        preds = []

        for x in X:
            posteriors = []
            for class_value in self.classes:
                em_X1, em_X2 = self.gmm_params[class_value]
                likelihood = self._calculate_likelihood(x, em_X1, em_X2)
                posteriors.append(likelihood * self.priors[class_value])

            preds.append(np.argmax(posteriors))

        return np.asarray(preds)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

#TODO: Fix function
def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    ###########################################################################
    # Logistic Regression
    logistic_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    logistic_model.fit(x_train, y_train)

    y_pred_train_lor = logistic_model.predict(x_train)
    y_pred_test_lor = logistic_model.predict(x_test)

    lor_train_acc = np.mean(y_train == y_pred_train_lor)
    lor_test_acc = np.mean(y_test == y_pred_test_lor)

    # Naive Bayes Gaussian
    nb_gaussian = NaiveBayesGaussian(k=k)
    nb_gaussian.fit(x_train, y_train)

    y_pred_train_bayes = nb_gaussian.predict(x_train)
    y_pred_test_bayes = nb_gaussian.predict(x_test)

    bayes_train_acc = np.mean(y_train == y_pred_train_bayes)
    bayes_test_acc = np.mean(y_test == y_pred_test_bayes)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

#TODO: Fix function
def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    ###########################################################################
    np.random.seed(1)

    # Define means and covariance matrices for the classes
    mu_a = np.array([0, 0])
    cov_a = np.array([[1, 0], [0, 1]])

    mu_b = np.array([3, 3])
    cov_b = np.array([[1, 0], [0, 1]])

    # Generate the data for the classes
    dataset_a_features = np.random.multivariate_normal(mu_a, cov_a, 200)
    dataset_a_labels = np.zeros(200, dtype=int)

    dataset_b_features = np.random.multivariate_normal(mu_b, cov_b, 200)
    dataset_b_labels = np.ones(200, dtype=int)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }