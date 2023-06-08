import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

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
        self.costs = []
        self.J_history = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        ###########################################################################
        self.weights = np.ones(self.k) / self.k

        # random initialization strategy: divide the data into k partitions
        data_split = np.array_split(data, self.k)

        # calculate the mus and the sigmas based on each partition

        self.mus = np.array([np.mean(partition) for partition in data_split])
        self.sigmas = np.array([np.std(partition) for partition in data_split])

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        ###########################################################################
        normalized_responses = [None] * self.k
        sum = 0
        for j in range(self.k):
            normal_vals = norm_pdf(data, self.mus[j], self.sigmas[j])
            calc_weights = self.weights[j] * normal_vals
            sum += calc_weights

            normalized_responses[j] = calc_weights

        normalized_responses /= sum

        return np.array(normalized_responses)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function calculates and updates the model parameters
        """
        # Obtain the size of  responsibilities
        size = data.shape[0]

        # hold the new mean and standard deviation values
        updated_weights = np.zeros(self.k)
        updated_mus = np.zeros(self.k)
        updated_sigmas = np.zeros(self.k)

        for i in range(self.k):
            # 1. calculate the updated weights
            updated_weights[i] = np.sum(self.responsibilities[i]) / size

            # 2. calculate the updated mus
            updated_mus[i] = np.sum(data * self.responsibilities[i]) / (updated_weights[i] * size)

            # 3. calculate the updated sigmas
            updated_sigmas[i] = np.sqrt(np.sum(self.responsibilities[i] * (data - updated_mus[i]) ** 2) / (
                    updated_weights[i] * size))

        return updated_weights, updated_mus, updated_sigmas

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def cost_function(self, data):
        sum_pdfs = np.zeros_like(data)

        for k in range(self.k):
            sum_pdfs += self.weights[k] * norm_pdf(data, self.mus[k], self.sigmas[k])

        sum_cost = 0
        for k in range(self.k):
            sum_cost += (-np.log(np.sum(sum_pdfs)))

        return sum_cost


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
        current_costs = []

        self.init_params(data)

        # Perform the iterations

        for i in range(self.n_iter):
            # 1. E-step
            self.responsibilities = self.expectation(data)

            #2. M-step
            self.weights, self.mus, self.sigmas = self.maximization(data)

            # Calculate the cost for each cluster
            current_cost = self.cost_function(data)
            current_costs.append(current_cost)

            if len(current_costs) > 1 and abs(current_costs[-2] - current_cost) <= self.eps:
                break

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
    pdf = np.sum(norm_pdf(data, mus, sigmas) * weights)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

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

            first_feat = EM(k=self.k)
            second_feat = EM(k=self.k)

            first_feat.fit(class_data[:, 0])
            second_feat.fit(class_data[:, 1])

            self.gmm_params[class_value] = (first_feat, second_feat)

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

        ###########################################################################
        ###########################################################################
        preds = []

        for instance in X:
            posterior_probs = []
            for class_label in self.classes:
                first_feature_gmm, second_feature_gmm = self.gmm_params[class_label]

                likelihood_first_feature = gmm_pdf(instance[0], first_feature_gmm.weights, first_feature_gmm.mus,
                                                   first_feature_gmm.sigmas)

                likelihood_second_feature = gmm_pdf(instance[1], second_feature_gmm.weights, second_feature_gmm.mus,
                                                    second_feature_gmm.sigmas)

                # assuming feature independence
                total_likelihood = likelihood_first_feature * likelihood_second_feature

                posterior_probs.append(total_likelihood * self.priors[class_label])

            preds.append(np.argmax(posterior_probs))

        return np.asarray(preds)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = np.array(Z)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()
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

    # Plotting the Logistic Regression
    plt.figure()
    plot_decision_regions(x_train, y_train, classifier=logistic_model, title="Logistic Regression Decision Boundaries")
    print("Explanation of Graph 1: The area of each color in the graph represents "
          "the region where the Logstic Regression model predicts a class. "
          "Thus we can observe how well the model separates the different classes in the training set")

    # Naive Bayes Gaussian
    nb_gaussian = NaiveBayesGaussian(k=k)
    nb_gaussian.fit(x_train, y_train)

    y_pred_train_bayes = nb_gaussian.predict(x_train)
    y_pred_test_bayes = nb_gaussian.predict(x_test)

    bayes_train_acc = np.mean(y_train == y_pred_train_bayes)
    bayes_test_acc = np.mean(y_test == y_pred_test_bayes)

    # Plotting the Naive Bayes Gaussian
    plt.figure()
    plot_decision_regions(x_train, y_train, classifier=nb_gaussian, title="Naive Bayes Gaussian Decision Boundaries")
    print("Explanation of Graph 2: The area of each color in the graph represents "
          "the region where the Naive Bayes Gaussian model predicts a class. "
          "Thus we can observe how well the model separates the different classes in the training set")

    # Plotting cost Vs  iteration for Logistic Regression
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(logistic_model.Js)), logistic_model.Js)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations for Logistic Regression Model')
    plt.grid(True)
    plt.show()

    print("Explanation of Graph 3: the graph shows the Logistic Regression learning process."
          " As the curve is decreasing we see that the model is learning from the data."
          "As the slope of the curve is smaller, the model near convergence - the error is reduced "
          "in smaller amounts in each iteration. "
          )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


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
    mu_a = np.array([0, 0, 0])
    cov_a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    samples_a = multivariate_normal(mu_a, cov_a).rvs(200)
    labels_a = np.zeros(200, dtype=int)

    mu_b = np.array([3, 3, 3])
    cov_b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    samples_b = multivariate_normal(mu_b, cov_b).rvs(200)
    labels_b = np.ones(200)

    # Generate the data for the classes
    dataset_a_features = np.concatenate((samples_a, samples_b))
    dataset_a_labels = np.concatenate((labels_a,labels_b))

    mu_a = np.array([1, 1, 1])
    cov_a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    samples_a = multivariate_normal(mu_a, cov_a).rvs(200)
    labels_a = np.zeros(200, dtype=int)

    mu_b = np.array([4, 4, 4])
    cov_b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    samples_b = multivariate_normal(mu_b, cov_b).rvs(200)
    labels_b = np.ones(200)

    # Generate the data for the classes
    dataset_b_features = np.concatenate((samples_a, samples_b))
    dataset_b_labels = np.concatenate((labels_a, labels_b))



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }