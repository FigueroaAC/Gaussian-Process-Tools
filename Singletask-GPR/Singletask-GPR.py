import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import fmin_l_bfgs_b
from tqdm import tqdm
import Kernels
# In[]
# In[]
class Singletask:
    def __init__(self, x: np.ndarray, y: np.ndarray, size: int = 200, kernel_type: str = 'asqe',
                 n_opt: int = 200, n_classes: int = None, plot: bool = False):
        self.__x = x
        self.__y = y
        self.__size = size
        self.__kernel_type = kernel_type.lower()
        self.__n_opt = n_opt
        self.__n_classes = n_classes
        self.__kernel_n_params = 0
        self._kernel_opt_params = None
        self.__kernel = self.__get_kernel()
        self._Kernel = None
        self.__L_ = None
        self.__L_inv = None
        self.__alpha_ = None
        self.__K_inv = None
        self.__plot = plot

    def __get_kernel(self):
        kernels = {'sqe': Kernels.SQEKernel, 'asqe': Kernels.ASQEKernel, 'lap': Kernels.LAPKernel,
                   'alap': Kernels.ALAPKernel, 'linear': Kernels.LinearKernel, 'poly': Kernels.PolyKernel,
                   'anova': Kernels.AnovaKernel, 'sigmoid': Kernels.SigmoidKernel, 'rq': Kernels.RQKernel,
                   'multiquad': Kernels.MultiQuadKernel, 'invmultiquad': Kernels.InvMultiQuadKernel,
                   'power': Kernels.PowerKernel, 'wave': Kernels.WaveKernel, 'cauchy': Kernels.CauchyKernel,
                   'log': Kernels.LogKernel, 'tstudent': Kernels.TstudentKernel, 'srq': Kernels.SRQKernel,
                    'anovalinear': Kernels.AnovaLinearKernel, 'anovaasqe': Kernels.AnovaASQEKernel}
        return kernels[self.__kernel_type]

    def __get_n_params(self) -> int:
        v = self.__x
        n_params = {'sqe': 3, 'asqe': v.shape[-1] + 2, 'lap': 3, 'alap': v.shape[-1] + 2,
                    'linear': v.shape[-1] + 3, 'poly': v.shape[-1] + 4, 'anova': 3, 'sigmoid': 3,
                    'rq': v.shape[-1] + 2, 'multiquad': v.shape[-1] + 2 , 'invmultiquad': v.shape[-1] + 2,
                    'power': v.shape[1] + 2, 'wave': 2, 'cauchy': 2, 'log': 2, 'tstudent': v.shape[-1] + 2,
                    'srq': v.shape[-1] + 2, 'anovalinear': v.shape[-1] + 1, 'anovaasqe': (2*v.shape[-1]) + 1
                    }
        return n_params[self.__kernel_type]

    @staticmethod
    def __normalize_uniform_01(v: np.ndarray) -> (float, np.ndarray):

        mx = np.max(v, axis=0)
        vnorm = v/mx
        return vnorm, mx

    @staticmethod
    def __normalize_normal(v: np.ndarray) -> (float, float, np.ndarray):

        mean = np.mean(v)
        std = np.std(v)
        vnorm = (v-mean)/std

        return vnorm, mean, std

    @staticmethod
    def __revert_normalize_normal(vnorm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (vnorm*std) + mean

    def __split_train_test_sobol(self):

        x_train = self.__x[:self.__size]
        x_test = self.__x[self.__size:]

        y_train = self.__y[:self.__size]
        y_test = self.__y[self.__size:]

        return x_train, y_train, x_test, y_test

    def __preprocess_training_test_data(self):
        try:
            if self.__x.shape[1] >= 1:
                pass
        except IndexError:
            self.__x = self.__x.reshape(-1, 1)

        self.__x, self.__max_x = self.__normalize_uniform_01(self.__x)

        self.__x_train, self.__y_train, self.__x_test, self.__y_test = self.__split_train_test_sobol()
        self.__y_train, self.__ytr_mean, self.__ytr_std = self.__normalize_normal(self.__y_train)

    def __nll_fn(self):
        '''
        Returns a function that computes the negative params log marginal
        likelihood for training data X_train and Y_train w.r.t kernel parameters

        Args:
            X_train: training locations. (m x d) array | (d = n-dimensions)
            Y_train: training targets. (m x 1) array
            Type: Type of Kernel to use. String

        Returns:
            Minimization objective, gradients w.r.t kernel parameters
        '''

        def nll_stable(theta):
            '''

            Parameters
            ----------
            theta : array (1xn_params)
                Array containing proposals for hyperparameters.

            Returns
            -------
            nll: float
                negative loglikelihood of model.
            dnll: array of floats (1xn_params)
                gradient of the negative loglikehood w.r.t hyperparameters.

            '''
            # Numerical Implementation of Eq. (7) as described
            # in http://www.gaussianprocess.org/gpm/chapters/RW2.pdf, Section
            # 2.2, Algorithm 2.1.
            self._Kernel = self.__kernel(x1=self.__x_train, x2=self.__x_train, params=theta, with_gradient=True)
            K = self._Kernel.K
            dK = self._Kernel.Kgrad
            try:
                self.__L_ = linalg.cholesky(K, lower=True)
                self.__L_inv_ = linalg.solve_triangular(self.__L_.T, np.eye(self.__L_.shape[0]))
                self.__K_inv = self.__L_inv_.dot(self.__L_inv_.T)

                self.__alpha_ = linalg.cho_solve((self.__L_, True), self.__y_train)

                nll = np.sum(np.log(np.diagonal(self.__L_))) + \
                      0.5 * np.dot(self.__y_train.T, self.__alpha_) + \
                      0.5 * len(self.__x_train) * np.log(2 * np.pi)

                Tr_arg = self.__alpha_.dot(self.__alpha_.T) - self.__K_inv

                dnll = [-0.5 * np.trace(Tr_arg.dot(dK[i])) \
                        for i in range(theta.shape[0])]
                dnll = np.array(dnll)

                return nll , dnll

            except (np.linalg.LinAlgError, ValueError):
                # In case K is not positive semidefinite
                return np.inf , np.array([np.inf for i in range(theta.shape[0])])

        return nll_stable

    def __optimize(self) -> np.ndarray[float]:

        Output = []
        if self.__kernel_type == 'ASQE':
            for i in tqdm(range(self.__n_opt)):
                res = fmin_l_bfgs_b(self.__nll_fn(),
                                    np.random.random(self.__kernel_n_params), fprime=None,
                                    bounds=[(0, np.inf) \
                                            for j in range(self.__kernel_n_params - 1)] + \
                                           [(1e-12, 1e-8)])
                Output.append([res[0], res[1]])
        else:
            for i in tqdm(range(self.__n_opt)):
                res = fmin_l_bfgs_b(self.__nll_fn(),
                                    np.random.random(self.__kernel_n_params), fprime=None,
                                    bounds=[(0, np.inf) for j in range(self.__kernel_n_params)])
                Output.append([res[0], res[1]])

        return np.array(Output)

    @staticmethod
    def __get_rsquared(vpred, vtest):

        def __get_sstot(v):
            vmean = np.mean(v)
            diff = np.array([(v_i - vmean) ** 2 for v_i in v])
            sstot = np.sum(diff)
            return sstot

        def __get_ssres(vp, vt):
            diff = np.array([(vp[i] - vt[i]) ** 2 for i in range(len(vt))])
            ssres = np.sum(diff)
            return ssres

        sstot = __get_sstot(vtest)
        ssres = __get_ssres(vpred, vtest)
        rsquared = 1 - (ssres / sstot)
        return rsquared

    def posterior_predictive(self, x_s: np.ndarray, params:np.ndarray) -> (float, float):
        '''
        Computes the sufficient statistics of the GP posterior predictive distribution
        from m training data X_train and Y_train and n new inputs X_s.

        Args:
            X_s: New input params locations (n x d).
            X_train: Training locations (m x d).
            Y_train: Training targets (m x 1).
            l: Kernel length parameter.
            sigma_f: Kernel] vertical variation parameter.
            sigma_y: Noise parameter.

        Returns:
            Posterior mean vector (n x d) and covariance matrix (n x n).
        '''

        K_s = self.__kernel(x1=x_s, x2=self.__x_train, params=params, with_gradient=False).K
        mu_s = np.dot(K_s.T, self.__alpha_)
        # This part can be commented out if a calculation of the posterior variance
        # is not desired or needed. It doubles the calculation time

        K_ss = self.__kernel(x1=x_s, x2=x_s, params=params, with_gradient=True).K
        # L_inv = linalg.solve_triangular(L_.T,np.eye(L_.shape[0]))
        # K_inv = L_inv.dot(L_inv.T)
        var_s = np.diag(K_ss) - np.diag(K_s.T.dot(self.__K_inv.dot(K_s)))
        var_s = np.array([np.sqrt(variance) if variance > 0 else 0 for variance in var_s])

        return mu_s, var_s

    def __set_kernel_matrices(self, params: np.ndarray):
        self._Kernel = self.__kernel(x1=self.__x_train, x2=self.__x_train, params=params, with_gradient=False).K
        self.__L_ = linalg.cholesky(self._Kernel, lower=True)
        self.__alpha_ = linalg.cho_solve((self.__L_, True), self.__y_train)
        self.__L_inv = linalg.solve_triangular(self.__L_.T, np.eye(self.__L_.shape[0]))
        self.__K_inv = self.__L_inv.dot(self.__L_inv.T)

    def __get_kernel_matrices(self):
        # we check for those kernels where the inputs are divided by the parameter vector:
        inv_prop_params = ['asqe', 'sqe', 'alap', 'poly', 'rq', 'multiquad', 'invmultiquad',
                           'power', 'tstudent']

        if self.__kernel_type in inv_prop_params:
            self._kernel_opt_params[1:-1] = self._kernel_opt_params[1:-1] * self.__max_x
            self.__x_train = self.__x_train * self.__max_x
            self.__x_test = self.__x_test * self.__max_x

        try:
            r_squared = self.__compute_statistics(params=self._kernel_opt_params)
        except (ValueError, np.linalg.LinAlgError):
            print('The optimization did not converge to an invertible kernel')
            return {}

        Output = {'params': self._kernel_opt_params, 'alpha_': self.__alpha_, 'K_inv': self.__K_inv,
                  'r_squared': r_squared}

        if self.__kernel_type in inv_prop_params:
            self.__Lambda = np.eye(len(self.__x_train[0]))
            length_scales = 1/(self._kernel_opt_params[1:]**2)
            np.fill_diagonal(self.__Lambda, length_scales)
            Output['Lambda'] = self.__Lambda

        return Output

    def __make_predictions(self, params: np.ndarray):
        try:
            self.__set_kernel_matrices(params=params)
        except (ValueError, np.linalg.LinAlgError):
            return 0

        self.__mu_s, self.__std_s = self.posterior_predictive(self.__x_test, params=params)
        self.__mu_s = self.__revert_normalize_normal(self.__mu_s, self.__ytr_mean, self.__ytr_std)
        return self.__mu_s, self.__std_s

    def __compute_statistics(self, params: np.ndarray):
        try:
            mu_s, std = self.__make_predictions(params=params)
        except TypeError:
            return 0
        Error = self.__y_test - mu_s
        RMSE = np.sqrt(np.dot(Error, Error.T)) / len(Error)
        self.__rsq = self.__get_rsquared(mu_s, self.__y_test)
        return self.__rsq

    def plot_results(self):
        for i in range(self.__x_train.shape[1]):
            sort_idxs = np.argsort(self.__x_test[:,i])
            plt.figure()
            plt.suptitle('{}'.format(self.__kernel_type),fontweight='bold')

            if self.__n_classes is None:
                plt.scatter(self.__x_test[:, i][sort_idxs],
                            self.__mu_s[sort_idxs],
                            color='r', label='Reconstruction')
                plt.fill_between(self.__x_test[:, i][sort_idxs],
                                 self.__mu_s[sort_idxs] - self.__std_s[sort_idxs],
                                 self.__mu_s[sort_idxs] + self.__std_s[sort_idxs],
                                 alpha=0.3)
                # plt.scatter(X_train[:,i],Y_train, label='Train Set')
                plt.scatter(self.__x_test[:, i][sort_idxs], self.__y_test[sort_idxs],
                            label='Test Set', alpha=0.3)
                plt.grid(True)
                plt.legend()
                plt.show()

            else:
                for j in range(1, self.__n_classes+1):
                    m_idx = np.where(self.__x_test[sort_idxs][:, -j] == 1)
                    plt.scatter(self.__x_test[:, i][sort_idxs][m_idx],
                                self.__mu_s[sort_idxs][m_idx],
                                label='Class {} Reconstruction'.format(j))
                    plt.fill_between(self.__x_test[:, i][sort_idxs][m_idx],
                                     self.__mu_s[sort_idxs][m_idx] - self.__std_s[sort_idxs][m_idx],
                                     self.__mu_s[sort_idxs][m_idx] + self.__std_s[sort_idxs][m_idx],
                                     alpha=0.3)
                         #plt.scatter(X_train[:,i],Y_train,label='Train Set')
                    plt.scatter(self.__x_test[:, i][sort_idxs], self.__y_test[sort_idxs],
                                label='Test Set', alpha=0.3)
                    plt.grid(True)
                    plt.legend()
                    plt.show()

    def perform_gpr(self):
        self.__preprocess_training_test_data()
        self.__kernel_n_params = self.__get_n_params()
        print('\n Optimizing Parameters')
        opt_results = self.__optimize()
        print('\n Calculating Best Parameter Set')
        r_squared = np.array([self.__compute_statistics(params[0]) for params in tqdm(opt_results)])
        if not np.all(r_squared):
            print('The optimization did not converge to an invertible kernel')
            return {'r_squared': -np.inf, 'params': None}
        idx = np.argmax(np.array(r_squared))
        print(f'Optimal r_2 = {max(r_squared)}')
        print(f'Optimal parameters = {opt_results[idx][0]}')
        self._kernel_opt_params = opt_results[idx][0]
        kernel_dict = self.__get_kernel_matrices()
        if self.__plot:
            self.plot_results()
        return kernel_dict

# In[]
# =============================================================================
# Load input and outputs datasets, this here is an example
# =============================================================================

# =============================================================================
# Single Input Gaussian Process:
# =============================================================================


def test_single_input(kerneltype):
    def fsi(x):
        return x * np.sin(x)

    N_samples = 100
    X = 2*np.pi*np.random.random(N_samples)
    Y = fsi(X)
    # Verify the inputs and outputs shapes match the expectation
    print(X.shape)
    print(Y.shape)

    Gp = Singletask(x=X, y=Y, size=25, kernel_type=kerneltype, n_opt=1000, plot=True)
    res = Gp.perform_gpr()
    print(res['r_squared'])
    print(res['params'])
# In[]

# =============================================================================
# Multiple Input Gaussian Process:
# =============================================================================
def test_multiple_input(kerneltype):
    def fmi(x):
        return x[:, 0] * np.sin(x[:, 1]) + x[:, 2]

    N_samples = 500
    x1 = np.expand_dims(np.pi*np.random.random(N_samples), axis=-1)
    x2 = np.expand_dims(np.pi*np.random.random(N_samples), axis=-1)
    x3 = np.expand_dims(np.pi*np.random.random(N_samples), axis=-1)
    X = np.concatenate((x1, x2, x3), axis=-1)
    Y = fmi(X)
    # Verify the inputs and outputs shapes match the expectation
    print(X.shape)
    print(Y.shape)
    Gp = Singletask(x=X, y=Y, size=25, kernel_type=kerneltype, n_opt=1000, plot=True)
    res = Gp.perform_gpr()
    print(res['r_squared'])
    print(res['params'])
# In[]
# =============================================================================
# Multiclass & Multi-input Gaussian Process:
# =============================================================================


def test_multiclass_multipleinput(kerneltype):
    def f(x):
        Output = []
        for i in range(x.shape[0]):
            if x[i][2] == 1 :
                Output.append(x[i][0] * np.sin(x[i][1]))
            else:
                Output.append(np.sin(x[i][0]) + np.sin(x[i][1]))
        return np.array(Output)

    def to_categorical(x):
        n_classes = max(x)
        out = [([0]*x_i)+[1]+([0]*(n_classes-x_i)) for x_i in x]
        return np.array(out)

    Classes = np.array([0, 1])
    N_samples = 1000
    x1 = np.expand_dims(np.pi*np.random.random(N_samples), axis=-1)
    x2 = np.expand_dims(np.pi*np.random.random(N_samples), axis=-1)
    x3 = np.random.choice(Classes, size=N_samples)
    x3 = to_categorical(x3)
    X = np.concatenate((x1, x2, x3), axis=-1)

    Y = f(X)
    print(X.shape)
    print(Y.shape)

    plt.figure()
    plt.scatter(X[:, 0][np.where(X[:, -1] == 1)], Y[np.where(X[:, -1] == 1)], label='0')
    plt.scatter(X[:, 0][np.where(X[:, -2] == 1)], Y[np.where(X[:, -2] == 1)], label='1')
    plt.grid()
    plt.legend()
    Gp = Singletask(x=X, y=Y, size=25, kernel_type=kerneltype, n_opt=1000, plot=True, n_classes=len(Classes))
    res = Gp.perform_gpr()
    print(res['r_squared'])

# log, tstudent, srq are broken

test_multiple_input('ASQE')

# In[]
Data = np.loadtxt('',skiprows=1,
                  delimiter=',',usecols=(1,2,3,4,5,6,7,8,))
x = Data[:,:-1]
y = Data[:,-1]
# In[]

Gp = Singletask(x=x, y=y, size=120, kernel_type='linear', n_opt=1000, plot=True)
res = Gp.perform_gpr()