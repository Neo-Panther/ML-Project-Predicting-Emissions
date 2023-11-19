import numpy as np
import random
from datetime import datetime
import os
import json

class SVR():
    def __init__(self, kernel='rbf', C=1, gamma=1, 
        epsilon=0.1, random_seed=0, max_iter=-1, E_toler=0.001, debug=False):
        np.random.seed(random_seed)
        random.seed(random_seed)
        self._kernel = kernel
        self._C = C
        self._gamma = gamma
        self._epsilon = epsilon
        self._max_iter = max_iter
        self._E_toler = E_toler
        self._debug = debug

        self._is_fit = False
        self._custom_kernel = False
        self._all_KKT_passed = False

        # Check if kernel is custom function.
        if hasattr(self._kernel, '__call__'):
            self._custom_kernel = True
            assert self._check_mercer_s_conditions(), 'Kernel must satisfy Mercer\'s conditions.'

    def get_params(self):
        if self._custom_kernel:
            str_kernel = 'Custom Kernel'
        else:
            str_kernel = self._kernel
        print('kernel = {:}, C = {:}, gamma = {:}, epsilon = {:}, max_iter = {:}, debug = {:}'\
            .format(str_kernel, self._C, self._gamma, self._epsilon, self._max_iter, self._debug))

    def fit(self, train_X, train_y):
        # -----------------------------------------------------------------------------------------
        # Checking parameters.
        # Check train_X
        assert type(train_X) in [list, np.ndarray], 'Training set X must be numpy.ndarray or list.'
        train_X = np.array(train_X)
        train_X = self._check_X(train_X, label='train_X')

        # Check train_y
        assert type(train_y) in [list, np.ndarray], 'Training set y must be numpy.ndarray or list.'
        train_y = np.array(train_y)
        assert len(train_y.shape) == 1, 'Training set y must be 1-D instead of {:}-D'.format(len(train_y.shape))

        # Check data length
        assert train_X.shape[0] == train_y.shape[0], 'Training set X must have same length as Training set y.'
        
        self._is_fit = True
        self._train_X = train_X
        self._train_y = train_y

        # -----------------------------------------------------------------------------------------
        # Initialize training
        self._N, self._N_features = self._train_X.shape
        self.kerneled_matrix = self._kernel_trans(self._train_X, self._train_X)
        self._debug_print('Training data transfer to:')
        self._debug_print(self.kerneled_matrix)
        self._is_changed = np.zeros(self._N)
        self.alphas = np.zeros(self._N)
        self.b = 0

        # -----------------------------------------------------------------------------------------
        # Training with SMO
        iter_num = 1
        alphas_pairs_changed = 0

        while alphas_pairs_changed > 0 or iter_num == 1:
            alphas_pairs_changed = 0
            self._debug_print('=====================================')
            self._debug_print('iter_num =', iter_num)

            if iter_num == 1:
                seq_SV = list(range(self._N))
            else:
                seq_SV = np.nonzero((self.alphas > -self._C)*(self.alphas < self._C))[0]
                self._debug_print('non_bound =')
                self._debug_print(seq_SV)

            for i in seq_SV:
                self._debug_print('-------------------------------')
                self._debug_print('i =', i)
                alphas_pairs_changed += self._update_alphas(i)
                if self._all_KKT_passed:
                    break

            iter_num += 1
            if self._max_iter != -1 and iter_num > self._max_iter:
                break

        return self

    def predict(self, test_X):
        # -----------------------------------------------------------------------------------------
        # Checking parameters.
        # Check test_X
        assert type(test_X) in [list, np.ndarray], 'Testing set X must be numpy.ndarray or list.'
        test_X = np.array(test_X)
        test_X = self._check_X(test_X, label='test_X')

        # Check if it's already fit.
        assert self._is_fit, 'SVR model hasn\'t been trained.'

        # -----------------------------------------------------------------------------------------
        # Predict
        kerneled_test = self._kernel_trans(self._train_X, test_X)
        pred_y = np.dot(kerneled_test, self.alphas) + self.b

        return pred_y

    def save_model(self, save_dir='default', file_name='default'):
        # Check if model is trained.
        assert self._is_fit, 'SVR model hasn\'t been trained.'
        assert not self._custom_kernel, 'Custom kernel isn\'t supported for saveing model.'
        # Generate file dir.
        if save_dir == 'default':
            save_dir = 'trained_models/'
        if file_name == 'default':
            file_name = datetime.now().strftime('SVR_%Y%m%d_%H%M%S.txt')
        self._check_and_create_dir(save_dir)

        dejson_data = {
            'C': self._C,
            'gamma': self._gamma,
            'epsilon': self._epsilon,
            'max_iter': self._max_iter,
            'debug': self._debug,
            'kernel': self._kernel,
            'train_X': self._train_X.tolist(),
            'alphas': self.alphas.tolist(),
            'b': self.b
        }

        with open(save_dir + file_name, 'a', encoding='utf-8') as fp:
            fp.write(json.dumps(dejson_data))

        print('Model has been saved.')

    def load_model(self, file_dir):
        with open(file_dir, 'r+') as fp:
            json_data = fp.read()
        dejson_data = json.loads(json_data)

        self._C = dejson_data['C']
        self._gamma = dejson_data['gamma']
        self._epsilon = dejson_data['epsilon']
        self._max_iter = dejson_data['max_iter']
        self._debug = dejson_data['debug']
        self._kernel = dejson_data['kernel']
        self._train_X = np.array(dejson_data['train_X'])
        self.alphas = np.array(dejson_data['alphas'])
        self.b = dejson_data['b']

        self._is_fit = True

        print('Model has been loaded.')

        return self

    def _check_mercer_s_conditions(self):
        rand_X = np.random.random((5,2))
        trans_X = self._kernel_trans(rand_X, rand_X)
        n, m = train_X.shape

        condition_1 = n == m
        if not condition_1:
            return False

        condition_2 = True
        for i in range(n):
            for j in range(i+1, m):
                if trans_X[i, j] != trans_X[j, i]:
                    condition_2 = False
        condition_3 = np.all(np.linalg.eigvals(train_X) >= 0)

        return condition_1 and condition_2 and condition_3

    def _check_X(self, X, label):    
        if len(X.shape) == 1:
            print('Warning: {:} is 1-D, automatically reshape to ({:}, {:})'\
                .format(label, X.shape[0], 1))
            return X.reshape(-1, 1)
        elif len(X.shape) == 2:
            return X
        else:
            raise ValueError('{:} is {:}-D, expect 1-D or 2-D.'.format(label, len(X.shape)))

    def _kernel_trans(self, X, Y):
        if self._custom_kernel:
            return self.kernel(X, Y)
        elif self._kernel == 'linear':
            return Y.dot(X.T)
        elif self._kernel == 'rbf':
            K_rbf = []
            for y in Y:
                temp = np.exp(-np.sum(np.square(X - y), 1) / self._gamma**2)
                K_rbf.append(temp)
            return np.array(K_rbf)
        else:
            raise ValueError('The kernel: {:} isn\'t supported for now.'.format(self._kernel))

    def _update_alphas(self, i):
        self._debug_print('alphas = ')
        self._debug_print(self.alphas)
        self._debug_print('b = ')
        self._debug_print(self.b)

        # Calculate error at index i
        Ei = self._cal_error(i)
        self._debug_print('Ei =', Ei)

        # Check KKT condition
        is_violet_KKT = not self._check_KKT(i, Ei)
        if is_violet_KKT:
            self._debug_print('i doesn\'t violet the KKT condition.')
            return 0

        # Select j
        j, Ej = self._select_j(i, Ei)

        # If all of the SV has passed the KKT condition, break the training process.
        if j == -1:
            self._debug_print('all of the SV has passed the KKT condition.')
            self._all_KKT_passed = True
            return 0

        self._debug_print('j =', j)
        self._debug_print('Ej =', Ej)

        # Save the old i and j
        alpha_i_old = self.alphas[i].copy()
        alpha_j_old = self.alphas[j].copy()

        # Calculate the lower and upper bound
        lower_bound = max(-self._C, alpha_i_old + alpha_j_old - self._C)
        self._debug_print('lower_bound =', lower_bound)
        upper_bound = min(self._C, alpha_i_old + alpha_j_old + self._C)
        self._debug_print('upper_bound =', upper_bound)

        # This may not happen, but if the lower_bound equals to upper_bound, returning 0.
        if lower_bound == upper_bound:
            return 0

        # Calculate eta
        eta = self.kerneled_matrix[i, i] + self.kerneled_matrix[j, j] - 2.0 * self.kerneled_matrix[i, j]
        self._debug_print('eta =', eta)
        # May not happen
        if eta <= 0:
            self._debug_print('eta<=0')
            return 0

        # update j
        is_j_update = False
        I = alpha_i_old + alpha_j_old
        for sgn in [-2, 0, 2, -1, 1]:
            self._debug_print('try sgn =', sgn)
            temp_j = alpha_j_old + (Ei - Ej + self._epsilon)
            self._debug_print('temp_j =', temp_j)
            if np.sign(I - temp_j) - np.sign(temp_j) == sgn:
                self.alphas[j] = temp_j
                is_j_update = True
                break
        if not is_j_update:
            return 0

        self._debug_print('A_j,new =', self.alphas[j])
        self.alphas[j] = max(min(self.alphas[j], upper_bound), lower_bound)
        self._debug_print('A_j,new,cli =', self.alphas[j])

        if abs(self.alphas[j] - alpha_j_old) < 1e-5:
            self._debug_print('A_j changes too small.')
            self._is_changed[j] = 1
            return 0

        # update i
        self.alphas[i] += alpha_j_old - self.alphas[j]
        self._debug_print('A_i,new =', self.alphas[i])

        # Calculate bi and bj
        bi = -(Ei + (self.alphas[i] - alpha_i_old) * self.kerneled_matrix[i, i]\
            + (self.alphas[j] - alpha_j_old) * self.kerneled_matrix[i, j]) + self.b
        bj = -(Ei + (self.alphas[i] - alpha_i_old) * self.kerneled_matrix[i, j]\
            + (self.alphas[j] - alpha_j_old) * self.kerneled_matrix[j, j]) + self.b

        # Check if bi or bj is avaliable
        if abs(self.alphas[i]) < self._C:
            self.b = bi
            self._debug_print('bi is avaliable.')
        elif abs(self.alphas[j]) < self._C:
            self.b = bj
            self._debug_print('bj is avaliable.')
        else:
            self.b = (bi + bj) / 2
            self._debug_print('b update to (bi+bj)/2')

        self._is_changed[i] = 1
        self._is_changed[j] = 1

        return 1

    def _cal_error(self, i):
        fi = self.alphas.dot(self.kerneled_matrix[i].T) + self.b
        return fi - self._train_y[i]

    def _select_j(self, i, Ei):
        # i and j must be at least one violet the KKT condition.
        changed_seq = np.nonzero(self._is_changed)[0]
        self._debug_print('changed_seq =', changed_seq)
        if changed_seq.shape[0] == 0:
            random_seq = np.arange(self._N)
            random_seq = np.append(random_seq[:i], random_seq[i+1:])
            j = np.random.choice(random_seq)
            Ej = self._cal_error(j)
        else:
            max_step = -np.inf
            for c in changed_seq:
                if c == i:
                    continue
                temp_error = self._cal_error(c)
                temp_step = abs(temp_error - Ei)
                if temp_step > max_step:
                    max_step = temp_step
                    j, Ej = c, temp_error

        return j, Ej

    def _check_KKT(self, i, Ei=None):
        if isinstance(Ei, type(None)):
            Ei = self._cal_error(i)

        condition_1 = self.alphas[i] == 0 and abs(Ei) < self._epsilon + self._E_toler
        condition_2 = self.alphas[i] != 0 and abs(self.alphas[i]) < self._C and\
            abs(Ei) <= self._epsilon + self._E_toler and abs(Ei) >= self._epsilon - self._E_toler
        condition_3 = abs(self.alphas[i]) == self._C and abs(Ei) > self._epsilon - self._E_toler

        return not (condition_1 or condition_2 or condition_3)

    def _debug_print(self, *string):
        if self._debug:
            output_str = ''
            for s in string:
                output_str += self._var_form(s) + ' '
            print(output_str[:-1])

    def _var_form(self, var):
        # This method is made for making variable easily being copied to use in further coding.
        # check list, np.ndarray, tuple
        if type(var) in [list, np.ndarray, tuple]:
            if isinstance(var, list):
                temp_str, end_str = '[', ']'
            elif isinstance(var, np.ndarray):
                temp_str, end_str = 'np.array([', '])'
            elif isinstance(var, tuple):
                temp_str, end_str = '(', ')'

            if len(var) != 0:
                for v in var:
                    if isinstance(var, np.ndarray):
                        temp_str += self._var_form(v.tolist()) + ', '
                    else:
                        temp_str += self._var_form(v) + ', '
                temp_str = temp_str[:-2] + end_str
            else:
                temp_str += end_str

            return temp_str

        # check dict
        elif isinstance(var, dict):
            if len(var) != 0:
                temp_str = '{'
                for key, value in var.items():
                    temp_str += '{:}: {:}, '.format(self._var_form(key), self._var_form(value))
                temp_str = temp_str[:-2] + '}'
                return temp_str
            else:
                return '{}'
        else:
            return str(var)

    def _check_and_create_dir(self, detail_dir):
        arr_folder = detail_dir.split('/')
        current_dir = ''
        for f in arr_folder:
            current_dir += f
            if not os.path.isdir(current_dir):
                os.mkdir(current_dir)
            current_dir += '/'