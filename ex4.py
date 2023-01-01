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
        norm_2 = (np.linalg.norm([x - y], ord=2)) ** 2
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
        norm = np.linalg.norm([x - y], ord=1)
        exp_val = np.exp(-1 * beta * norm)
        return alpha * exp_val

    return kern


def Gibbs_kernel(alpha: float, beta: float, delta: float, gamma: float) -> Callable:
    """
    An implementation of the Gibbs kernel (see section 4.2.3 of http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """

    def l(x, beta, alpha, delta):
        norm_2 = np.linalg.norm([x - delta]) ** 2
        exp_val = np.exp(-1 * beta * norm_2)
        l = alpha * exp_val + gamma
        return l

    def gibbs_sqrt(l_x, l_y):
        no, de = 2 * l_x * l_y, ((l_x ** 2) + (l_y ** 2))
        return np.sqrt(no / de)

    def gibbs_exp_power(x, y, l_x, l_y):
        x_y_norm_2 = (np.linalg.norm([x - y])) ** 2
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
        product = np.inner(x,x)
        beta_2 = 2 * beta * (1 + product)
        return (1 + beta_2)

    def kern(x, y):
        one_2_beta_x, one_2_beta_y = one_2_beta(x, beta), one_2_beta(y, beta)
        product = np.inner(x,y)
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
        self.kernel = kernel
        self.noise = noise
        self.N = 0
        self.X = None
        self.K = None
        self.C = None
        self.y = None
        self.alpha = None

    def calculate_params(self):
        self.C = np.zeros((self.N, self.N))
        self.K = np.zeros((self.N, self.N))
        for i, x_i in enumerate(self.X):
            for j, x_j in enumerate(self.X):
                k = self.kernel(x_i, x_j)
                self.K[i, j] = k
                self.C[i, j] = k
                if i == j:
                    self.C[i, j] += self.noise

    def fit(self, X, y) -> 'GaussianProcess':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        self.y = y
        self.X = X
        self.N = len(self.X)
        self.calculate_params()

        cholesky = np.linalg.cholesky(self.C)
        self.L = cholesky
        cholesky_T = np.transpose(cholesky)
        # A\b = x WHICH SOLVES Ax=b
        xLy = np.linalg.solve(cholesky, y)
        self.alpha = np.linalg.solve(cholesky_T, xLy)
        return self

    def calculate_K_star(self, pX):
        n = len(pX)
        K_p = np.zeros((self.N, n))
        for i, x_i in enumerate(self.X):
            for j, z_j in enumerate(pX):
                K_p[i, j] = self.kernel(x_i, z_j)
        return K_p

    def calculate_C_z(self, X):
        n = len(X)
        gram = np.zeros((n, n))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                gram[i, j] = self.kernel(x_i, x_j)
        return gram

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the MMSE regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        K_s = self.calculate_K_star(X)
        return np.transpose(K_s) @ self.alpha

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
        K_star = self.calculate_K_star(X)
        K_T = np.transpose(K_star)
        C_z = self.calculate_C_z(X)

        # L\k_* = v which solves
        v = np.linalg.solve(self.L, K_star)
        pos_mean = K_T @ self.alpha
        pos_cov = C_z - (np.transpose(v) @ v)

        random_f = np.random.multivariate_normal(pos_mean, pos_cov)
        return random_f

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        K_star = self.calculate_K_star(X)
        C_z = self.calculate_C_z(X)

        # L\k_* = v which solves
        v = np.linalg.solve(self.L, K_star)
        pos_cov = C_z - (np.transpose(v) @ v)

        return np.array([np.sqrt(v_i) for v_i in pos_cov.diagonal()])

    def log_evidence(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the model's log-evidence under the training data
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the log-evidence of the model under the data points
        """
        self.fit(X, y)
        N = len(y)
        part_1 = -1 * 0.5 * np.transpose(y) @ self.alpha
        part_2 = -1 * np.sum([np.log(self.L[i, i]) for i in range(N)])
        part_3 = -1 * N * 0.5 * np.log(2 * np.pi)
        return part_1 + part_2 + part_3


def plot_prior(gp, X, p):
    plot_colors = ["teal", "darkslategrey", "cadetblue", "lightseagreen", "darkturquoise"]
    plt.figure()
    n = len(X)

    X_gram = np.array(gp.calculate_C_z(X))
    mu = np.zeros(n)
    ci = np.array([np.sqrt(k_x) for k_x in X_gram.diagonal()])
    plt.plot(X, mu, color="mediumturquoise", label='prior mean', linestyle='dashed')
    plt.fill_between(X, mu - ci, mu + ci, color="paleturquoise", alpha=.5, label='confidence interval')

    random_f = np.random.multivariate_normal(mu, X_gram, 5)
    for i, f in enumerate(random_f):
        plt.plot(X, f, color=plot_colors[i], label=f'function:{i + 1}')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title('prior: ' + KERNEL_STRS[p[0]].format(*p[2:]))
    plt.ylim([-5, 5])
    plt.legend()
    plt.show()


def main():
    # ------------------------------------------------------ section 2.1
    xx = np.linspace(-5, 5, 500)
    x, y = np.array([-2, -1, 0, 1, 2]), np.array([-2.1, -4.3, 0.7, 1.2, 3.9])

    # ------------------------------ questions 2 and 3
    # choose kernel parameters
    params = [
        # # Laplacian kernels
        ['Laplacian', Laplacian_kernel, 1, 0.5],  # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, 0.1, 0.5],  # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, 1, 1],  # insert your parameters, order: alpha, beta

        # RBF kernels
        ['RBF', RBF_kernel, 1, 0.25],  # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, 2, 0.25],  # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, 5, 0.5],  # insert your parameters, order: alpha, beta

        # # Gibbs kernels
        ['Gibbs', Gibbs_kernel, 5, 0.5, 2, .1],  # insert your parameters, order: alpha, beta, delta, gamma
        ['Gibbs', Gibbs_kernel, 4, 0.25, 0, 0.3],  # insert your parameters, order: alpha, beta, delta, gamma
        ['Gibbs', Gibbs_kernel, 6, 0.5, -1, 0.1],  # insert your parameters, order: alpha, beta, delta, gamma

        # Neurel network kernels
        ['NN', NN_kernel, 2, 0.25],  # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, 0.25, 1],  # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, 1, 1],  # insert your parameters, order: alpha, beta
    ]
    noise = 0.05

    # plot all of the chosen parameter settings
    for p in params:
        # create kernel according to parameters chosen
        k = p[1](*p[2:])  # p[1] is the kernel function while p[2:] are the kernel parameters

        # initialize GP with kernel defined above
        gp = GaussianProcess(k, noise)

        # plot prior variance and samples from the priors
        plot_prior(gp, xx, p)

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
        plt.title('posterior: ' + KERNEL_STRS[p[0]].format(*p[2:]))
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
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=min_ev), noise).fit(x, y).predict(xx), lw=2,
             label=f'min evidence beta:{round(min_ev, 2)}')
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=median_ev), noise).fit(x, y).predict(xx), lw=2,
             label=f'median evidence beta:{round(median_ev, 2)}')
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=max_ev), noise).fit(x, y).predict(xx), lw=2,
             label=f'max evidence beta:{round(max_ev, 2)}')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.title(f'GPs with best, mid and min log-evidence:')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
