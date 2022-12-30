import numpy as np
from typing import Callable
from matplotlib import pyplot as plt

KERNEL_STRS = {
    'Laplacian': r'Laplacian, $\alpha={}$, $\beta={}$',
    'RBF': r'RBF, $\alpha={}$, $\beta={}$',
    'Gibbs': r'Gibbs, $\alpha={}$, $\beta={}$, $\delta={}$, $\gamma={}$',
    'NN': r'NN, $\alpha={}$, $\beta={}$'
}


def average_error(pred: np.ndarray, vals: np.ndarray):
    """
    Calculates the average squared error of the given predictions
    :param pred: the predicted values
    :param vals: the true values
    :return: the average squared error between the predictions and the true values
    """
    return np.mean((pred - vals) ** 2)


def RBF_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the RBF kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """

    def kern(x, y):
        norm_2 = np.linalg.norm(x - y) ** 2
        exp_val = np.exp(-1 * beta * norm_2)
        return alpha * exp_val

    return kern


def Laplacian_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Laplacian kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """

    def kern(x, y):
        norm = np.linalg.norm(x - y, ord=1)
        exp_val = np.exp(-1 * beta * norm)
        return alpha * exp_val

    return kern


def Gibbs_kernel(alpha: float, beta: float, delta: float, gamma: float) -> Callable:
    """
    An implementation of the Gibbs kernel (see section 4.2.3 of http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """

    def l(x, beta, alpha, delta):
        norm_2 = np.linalg.norm(x - delta) ** 2
        exp_val = np.exp(-1 * beta * norm_2)
        l = alpha * exp_val
        return l

    def gibbs_sqrt(l_x, l_y):
        no, de = 2 * l_x * l_y, ((l_x ** 2) + (l_y ** 2))
        return np.sqrt(no / de)

    def gibbs_exp_power(x, y, l_x, l_y):
        x_y_norm_2 = (np.norm(x - y)) ** 2
        no, de = -1 * x_y_norm_2, ((l_x ** 2) + (l_y ** 2))
        return np.exp(no / de)

    def kern(x, y):
        l_x = l(x, beta, alpha, delta)
        l_y = l(y, beta, alpha, delta)
        sqrt_val = gibbs_sqrt(l_x, l_y)
        power_val = gibbs_exp_power(x, y, l_x, l_y)
        return sqrt_val * power_val

    return kern


def NN_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Neural Network kernel (see section 4.2.3 of http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """

    def one_2_beta(x, beta):
        product = np.inner(x)
        beta_2 = 2 * beta * (1 + product)
        return (1 + beta_2)

    def kern(x, y):
        one_2_beta_x, one_2_beta_y = one_2_beta(x, beta), one_2_beta(y, beta)
        product = np.inner(x, y)
        mul_1_2_betas = one_2_beta_x * one_2_beta_y
        no, de = 2 * beta * (product + 1), np.sqrt(mul_1_2_betas)
        arc_sin_val = np.arcsin(no / de)
        return alpha * (2 / np.pi) * arc_sin_val

    return kern


class GaussianProcess:

    def __init__(self, kernel: Callable, noise: float):
        """
        Initialize a GP model with the specified kernel and noise
        :param kernel: the kernel to use when fitting the data/predicting
        :param noise: the sample noise assumed to be added to the data
        """
        # todo <your code here>

    def fit(self, X, y) -> 'GaussianProcess':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        # todo <your code here>
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the MMSE regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        # todo <your code here>
        return None

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def sample(self, X) -> np.ndarray:
        """
        Sample a function from the posterior
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the sample (same shape as X)
        """
        # todo <your code here>
        return None

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        # todo <your code here>
        return None

    def log_evidence(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the model's log-evidence under the training data
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the log-evidence of the model under the data points
        """
        self.fit(X, y)
        # todo <your code here>
        return None


def main():
    # ------------------------------------------------------ section 2.1
    xx = np.linspace(-5, 5, 500)
    x, y = np.array([-2, -1, 0, 1, 2]), np.array([-2.1, -4.3, 0.7, 1.2, 3.9])

    # ------------------------------ questions 2 and 3
    # choose kernel parameters
    params = [
        # Laplacian kernels
        ['Laplacian', Laplacian_kernel, 1, 0.25],  # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, None, None],  # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, None, None],  # insert your parameters, order: alpha, beta

        # RBF kernels
        ['RBF', RBF_kernel, 1, 0.25],  # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, None, None],  # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, None, None],  # insert your parameters, order: alpha, beta

        # Gibbs kernels
        ['Gibbs', Gibbs_kernel, 1, 0.5, 0, .1],  # insert your parameters, order: alpha, beta, delta, gamma
        ['Gibbs', Gibbs_kernel, None, None, None, None],  # insert your parameters, order: alpha, beta, delta, gamma
        ['Gibbs', Gibbs_kernel, None, None, None, None],  # insert your parameters, order: alpha, beta, delta, gamma

        # Neurel network kernels
        ['NN', NN_kernel, 1, 0.25],  # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, None, None],  # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, None, None],  # insert your parameters, order: alpha, beta
    ]
    noise = 0.05

    # plot all of the chosen parameter settings
    for p in params:
        # create kernel according to parameters chosen
        k = p[1](*p[2:])  # p[1] is the kernel function while p[2:] are the kernel parameters

        # initialize GP with kernel defined above
        gp = GaussianProcess(k, noise)

        # plot prior variance and samples from the priors
        plt.figure()
        # todo <your code here>
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])

        # fit the GP to the data and calculate the posterior mean and confidence interval
        gp.fit(x, y)
        m, s = gp.predict(xx), 2 * gp.predict_std(xx)

        # plot posterior mean, confidence intervals and samples from the posterior
        plt.figure()
        plt.fill_between(xx, m - s, m + s, alpha=.3)
        plt.plot(xx, m, lw=2)
        for i in range(6): plt.plot(xx, gp.sample(xx), lw=1)
        plt.scatter(x, y, 30, 'k')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])
        plt.show()

    # ------------------------------ question 4
    # define range of betas
    betas = np.linspace(.1, 15, 101)
    noise = .15

    # calculate the evidence for each of the kernels
    evidence = [GaussianProcess(RBF_kernel(1, beta=b), noise).log_evidence(x, y) for b in betas]

    # plot the evidence as a function of beta
    plt.figure()
    plt.plot(betas, evidence, lw=2)
    plt.xlabel(r'$\beta$')
    plt.ylabel('log-evidence')
    plt.show()

    # extract betas that had the min, median and max evidence
    srt = np.argsort(evidence)
    min_ev, median_ev, max_ev = betas[srt[0]], betas[srt[(len(evidence) + 1) // 2]], betas[srt[-1]]

    # plot the mean of the posterior of a GP using the extracted betas on top of the data
    plt.figure()
    plt.scatter(x, y, 30, 'k', alpha=.5)
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=min_ev), noise).fit(x, y).predict(xx), lw=2, label='min evidence')
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=median_ev), noise).fit(x, y).predict(xx), lw=2,
             label='median evidence')
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=max_ev), noise).fit(x, y).predict(xx), lw=2, label='max evidence')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.legend()
    plt.show()

    # ------------------------------------------------------ section 2.2
    # define function and parameters
    f = lambda x: np.sin(x * 3) / 2 - np.abs(.75 * x) + 1
    xx = np.linspace(-3, 3, 100)
    noise = .25
    beta = 2

    # calculate the function values
    np.random.seed(0)
    y = f(xx) + np.sqrt(noise) * np.random.randn(len(xx))

    # ------------------------------ question 5
    # fit a GP model to the data
    gp = GaussianProcess(kernel=RBF_kernel(1, beta=beta), noise=noise).fit(xx, y)

    # calculate posterior mean and confidence interval
    m, s = gp.predict(xx), 2 * gp.predict_std(xx)
    print(f'Average squared error of the GP is: {average_error(m, y):.2f}')

    # plot the GP prediction and the data
    plt.figure()
    plt.fill_between(xx, m - s, m + s, alpha=.5)
    plt.plot(xx, m, lw=2)
    plt.scatter(xx, y, 30, 'k', alpha=.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim([-3, 3])
    plt.show()


if __name__ == '__main__':
    main()
