import numpy as np
import pandas as pd
import gc
from copy import copy
from tqdm import trange
from datetime import datetime
import warnings
from sklearn.metrics.pairwise import pairwise_kernels

class kernel_ic_maximizer:
    def __init__(self, characteristics, ret, starting_date, horizon = 10000, lag = [0], lamb = 1e-3,
            metric = 'linear', gamma = None, degree = None, coef0 = None
        ):
        self.characteristics = characteristics
        self.ret = ret
        self.starting_date = starting_date
        self.horizon = horizon
        self.lag = lag
        self.lamb = lamb
        self.metric = metric
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.date = list(characteristics.keys())
        
        
    def kernel_cov_maximizer(self, exog, endog, new_exog):
        gram_matrix = pairwise_kernels(
            new_exog, exog, metric = self.metric, filter_params = True,
            gamma = self.gamma, degree = self.degree, coef0 = self.coef0
        )
        gram_matrix = gram_matrix - gram_matrix.mean(axis = 0) - gram_matrix.mean(axis = 1).reshape(gram_matrix.shape[0], 1) + gram_matrix.mean()
        pred = gram_matrix.dot(endog)
        del gram_matrix
        gc.collect()

        return pred
    
    def ic_maximizer(self, exog, endog, new_exog):
        N1 = exog.shape[0]
        N2 = new_exog.shape[0]
        gram_matrix1 = pairwise_kernels(
            exog, exog, metric = self.metric, filter_params = True,
            gamma = self.gamma, degree = self.degree, coef0 = self.coef0
        )
        gram_matrix1 = gram_matrix1 - gram_matrix1.mean(axis = 0) - gram_matrix1.mean(axis = 1).reshape(N1, 1) + gram_matrix1.mean()
        gram_matrix2 = pairwise_kernels(
            new_exog, exog, metric = self.metric, filter_params = True,
            gamma = self.gamma, degree = self.degree, coef0 = self.coef0
        )
        gram_matrix2 = gram_matrix2 - gram_matrix2.mean(axis = 0) - gram_matrix2.mean(axis = 1).reshape(N2, 1) + gram_matrix2.mean()
        pred = gram_matrix2.dot(np.linalg.inv(gram_matrix1 + self.lamb * N1 * np.eye(N1))).dot(endog)
        del gram_matrix1, gram_matrix2
        gc.collect()

        return pred
    
    
    def rolling_test(self):
        saving_path = 'D:/kernel_prediction'
        prediction = 0 * self.ret

        oos_starting_date = self.date.index(self.starting_date)

        for t in trange(oos_starting_date, len(self.date)):
            new_exog = self.characteristics[self.date[t]]
            for i in self.lag:
                new_exog_ = self.characteristics[self.date[t - i]].add_suffix('_{}'.format(i))
                new_exog = new_exog.merge(new_exog_, how = 'left', left_index = True, right_index = True).fillna(0)
            prediction.loc[self.date[t]] = 0 * self.ret.loc[self.date[t]]
            new_available_list = list(new_exog.index)

            for s in range(max(t - self.horizon, max(self.lag)), t):
                exog = self.characteristics[self.date[s]]
                for i in self.lag:
                    exog_ = self.characteristics[self.date[s - i]].add_suffix('_{}'.format(i))
                    exog = exog.merge(exog_, how = 'left', left_index = True, right_index = True).fillna(0)
                available_list = list(exog.index)
                endog = self.ret.loc[self.date[s], available_list]
#                 pret = self.kernel_cov_maximizer(exog, endog, new_exog)
                pret = self.ic_maximizer(exog, endog, new_exog)
                pret_ = pd.DataFrame(pret.reshape(1, len(new_available_list)), columns = new_available_list)
                pret_.to_pickle(saving_path + '\{}-{}.pkl'.format(self.date[s], self.date[t]))
                prediction.loc[self.date[t], new_available_list] = prediction.loc[self.date[t], new_available_list] + pret

        pred = copy(prediction)
        pred[pred == 0] = np.nan
        ic = pred.iloc[oos_starting_date:].corrwith(self.ret.loc[pred.iloc[oos_starting_date:].index, pred.iloc[oos_starting_date:].columns], axis = 1).mean(skipna = True)
        print('IC : {}%'.format(100 * ic))
        return prediction