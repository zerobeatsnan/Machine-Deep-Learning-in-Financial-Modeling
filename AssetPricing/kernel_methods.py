import numpy as np
import pandas as pd
import gc
from copy import copy
from tqdm import trange
from datetime import datetime
import warnings
from sklearn.metrics.pairwise import pairwise_kernels
from joblib import Parallel, delayed
import pickle
import os


class AlwaysNoneList:
    def __getitem__(self, index):
        return None

class kernel_ic_maximizer:
    def __init__(self, characteristics, ret):
        # self.date = date
        self.ret = ret
        self.characteristics = characteristics
        self.date_ls = list(characteristics.keys())
        # self.reading_path = 'D:/cha'
        # self.prediction_saving_path = 'D:/kernel_prediction'

    def get_return_dict(self):

        real_return = {}

        for date in self.date_ls:

            real_ret = self.ret.loc[date, self.characteristics[date].index].values

            real_return[date] = real_ret

        self.return_dict = real_return


        
    
    def ic_maximizer(self, exog, endog, new_exog, lamb = 1e-3, metric = 'linear', gamma = None, degree = None, coef0 = None):
        N1 = exog.shape[0]
        N2 = new_exog.shape[0]
        gram_matrix1 = pairwise_kernels(
            exog, exog, metric = metric, filter_params = True,
            gamma = gamma, degree = degree, coef0 = coef0
        )
        gram_matrix1 = gram_matrix1 - gram_matrix1.mean(axis = 0) - gram_matrix1.mean(axis = 1).reshape(N1, 1) + gram_matrix1.mean()
        gram_matrix2 = pairwise_kernels(
            new_exog, exog, metric = metric, filter_params = True,
            gamma = gamma, degree = degree, coef0 = coef0
        )
        gram_matrix2 = gram_matrix2 - gram_matrix2.mean(axis = 0) - gram_matrix2.mean(axis = 1).reshape(N2, 1) + gram_matrix2.mean()
        pred = gram_matrix2.dot(np.linalg.inv(gram_matrix1 + lamb * N1 * np.eye(N1))).dot(endog)
        del gram_matrix1, gram_matrix2
        gc.collect()

        return pred
    
    def get_prediction_pair_of_lag_s(self, s = 1, lamb = 1e-3, metric = "linear", gamma = None, degree = None, coef0 = None):

        pred_return = {}

        real_return = {}

        for i in trange(len(self.date_ls[s:])):

            date = self.date_ls[s+i]
            
            data_pred = self.characteristics[date].values

            real_ret = self.ret.loc[date, self.characteristics[date].index].values


            index_pred = self.date_ls.index(date)

            index_train = index_pred - s

            date_train = self.date_ls[index_train]

            data_train = self.characteristics[date_train].values

            return_train = self.ret.loc[date_train, self.characteristics[date_train].index].values

            pred_ret = self.ic_maximizer(data_train, return_train, data_pred, lamb, metric, gamma, degree, coef0)

            pred_return[date] = pred_ret

            real_return[date] = real_ret

        return pred_return, real_return 
    
    def get_prediction_of_rolling_lag(self, lag_ls, weight_ls, lamb = 1e-3, metric = "linear",
                                      gamma = None, degree = None, coef0 = None):
        
        pred_return = {}

        max_lag = lag_ls[-1]

        for i in trange(len(self.date_ls[max_lag:])):

            date = self.date_ls[max_lag+i]

            data_pred = self.characteristics[date].values

            index_pred = self.date_ls.index(date)

            weighted_pred_ret = 0

            for j in range(len(lag_ls)):

                lag = lag_ls[j]

                weight = weight_ls[j]

                index_train = index_pred - lag
                
                date_train = self.date_ls[index_train]

                data_train = self.characteristics[date_train].values

                return_train = self.ret.loc[date_train, self.characteristics[date_train].index].values

                pred_ret = self.ic_maximizer(data_train, return_train, data_pred, lamb, metric, gamma, degree, coef0)

                weighted_pred_ret = weighted_pred_ret + weight * pred_ret

            pred_return[date] = weighted_pred_ret
            

        return pred_return






    
    def get_prediction_of_lag_s(self, s = 1, lamb = 1e-3, metric = "linear", gamma = None, degree = None, coef0 = None):

        pred_return = {}


        for i in trange(len(self.date_ls[s:])):

            date = self.date_ls[s+i]
            
            data_pred = self.characteristics[date].values

            index_pred = self.date_ls.index(date)

            index_train = index_pred - s

            date_train = self.date_ls[index_train]

            data_train = self.characteristics[date_train].values

            return_train = self.ret.loc[date_train, self.characteristics[date_train].index].values

            pred_ret = self.ic_maximizer(data_train, return_train, data_pred, lamb, metric, gamma, degree, coef0)

            pred_return[date] = pred_ret


        return pred_return
    
    def calculate_IC_of_lag_s(self, s = 1, lamb = 1e-3, metric = "linear", gamma = None, degree = None, coef0 = None):

        pred_return, real_return = self.get_prediction_pair_of_lag_s(s, lamb, metric, gamma, degree, coef0)

        correlations = []

        for key in pred_return.keys():

            corr = np.corrcoef(pred_return[key], real_return[key])[0,1]

            correlations.append(corr)

        avg_corr = np.mean(correlations)

        print(avg_corr)

        return avg_corr


    def cal_pred_dict(self, n_jobs, parallel = False, lag_ls = AlwaysNoneList(), lamb = 1e-3, metric = "linear", 
                      gamma_ls = AlwaysNoneList(), degree_ls = AlwaysNoneList(), coef_ls = AlwaysNoneList()):
        
        param_tuple = list(zip(lag_ls, gamma_ls, degree_ls, coef_ls))
        
        """
        the parameter_tuple is to acess the parameter used in each lag
        
        for lag s, to acess:
        
        lag: param_tuple[s-1][0]
        
        gamma: param_tuple[s-1][1]

        degree: param_tuple[s-1][2]

        coef: param_tuple[s-1][3]
        
        """

        if parallel:

            results = Parallel(n_jobs=n_jobs)(
                delayed(self.get_prediction_of_lag_s)(s = tup[0], lamb = lamb, metric = metric, gamma = tup[1], degree = tup[2], coef0 = tup[3])
                for tup in param_tuple
            )

            results_dict = {s:dict for (s,dict) in zip(lag_ls, results)}

            target_folder = "data/saved_data"

            file_path = os.path.join(target_folder, "dictionary for prediction dictionary".pkl)

            with open(file_path, "wb") as file:

                pickle.dump(results_dict, file)

            print(f"Dictionary save to {file=} successfully")

            return results_dict
        
        else: 

            results = []

            for i in range(len(param_tuple)):

                tup = param_tuple[i]

                results.append(self.get_prediction_of_lag_s(s = tup[0], lamb = lamb, metric = metric, gamma = tup[1], degree = tup[2], coef0 = tup[3]))


            results_dict = {s:dict for (s,dict) in zip(lag_ls, results)}

            target_folder = "data/saved_data"

            file_path = os.path.join(target_folder, "dictionary for prediction dictionary.pkl")

            with open(file_path, "wb") as file:

                pickle.dump(results_dict, file)

            print(f"Dictionary save to {file=} successfully")

            return results_dict
        
        







    

    
    
    # def get_prediction_of_lag_s(self, s = 1, lamb = 1e-3, metric = 'linear', gamma = None, degree = None, coef0 = None):
    #     prediction = 0 * self.ret

    #     oos_starting_date = self.date.index(self.starting_date)
    #     for t in trange(max(s, oos_starting_date), len(self.date)):
    #         key1 = self.date[t]
    #         key2 = self.date[t - s]
    #         new_exog = pd.read_pickle(self.reading_path + '/{}.pkl'.format(key1))
    #         new_available_list = list(new_exog.index)
    #         exog = pd.read_pickle(self.reading_path + '/{}.pkl'.format(key2))
    #         available_list = list(exog.index)
    #         endog = self.ret.loc[key2, available_list]
    #         prediction.loc[key1, new_available_list] = self.ic_maximizer(
    #             exog, endog, new_exog, lamb, metric, gamma, degree, coef0
    #         )
            
    #     pre = copy(prediction)
    #     pre[pre == 0] = np.nan
    #     ic = pre.corrwith(ret.loc[pre.index, pre.columns], axis = 1).mean()
    #     print(ic)
    #     saving_name = 's = {}, lamb = {}, metric = {}'.format(s, lamb, metric)
    #     if gamma is not None:
    #         saving_name = saving_name + ', gamma = {}'.format(gamma)
    #     if degree is not None:
    #         saving_name = saving_name + ', degree = {}'.format(degree)
    #     if coef0 is not None:
    #         saving_name = saving_name + ', coef0 = {}'.format(coef0)
    #     saving_name = saving_name + '.pkl'
    #     prediction.to_pickle(self.prediction_saving_path + '/' + saving_name)

def run_parallel_cal_IC(characteristics, ret, n_jobs, kernel_method, time_lags, gamma_range):

    model = kernel_ic_maximizer(characteristics, ret)

    results = Parallel(n_jobs = n_jobs)(
        delayed(model.calculate_IC_of_lag_s)(s, 1e-3, kernel_method, gamma, None, None )
        for s in time_lags for gamma in gamma_range
    )

    results_dict = {(s,gamma): ic for (s, gamma), ic in zip([(s, gamma) for s in time_lags for gamma in gamma_range], results)}

    target_folder = "data/saved_data"

    file_path = os.path.join(target_folder, "IC dictionary for different gamma and lag.pkl")

    with open(file_path, "wb") as file:

        pickle.dump(results_dict, file)

    print(f"Dictionary save to {file=} successfully")

    return results_dict



def cov_ls(dict_of_dict, target_dict):

    time_lags = list(dict_of_dict.keys())

    exp_ls = np.zeros((len(time_lags), 1))

    for i in range(len(time_lags)):

        time_lag = time_lags[i]

        exp_ls[i,0] = calculate_cov(dict_of_dict[time_lag], target_dict)

    return exp_ls

def cor_ls(dict_of_dict, target_dict):

    time_lags = list(dict_of_dict.keys())

    exp_ls = np.zeros((len(time_lags), 1))

    for i in range(len(time_lags)):

        time_lag = time_lags[i]

        exp_ls[i,0] = IC_of_two_dict(dict_of_dict[time_lag], target_dict)

    return exp_ls

    



def calculate_cov(timelag1, timelag2):

    overlapping_years = set(timelag1.keys()) & set(timelag2.keys())

    covariances = [np.cov(timelag1[year], timelag2[year])[0,1] for year in overlapping_years]

    return np.mean(covariances) if covariances else 0



def cov_mat(dict_of_dict):

    time_lags = list(dict_of_dict.keys())

    cov_matrix = np.zeros((len(time_lags), len(time_lags)))

    for i in range(len(time_lags)):

        for j in range(len(time_lags)):

            cov_matrix[i,j] = calculate_cov(dict_of_dict[time_lags[i]], dict_of_dict[time_lags[j]])

    return cov_matrix

def IC_of_two_dict(dict_1, dict_2):

    overlapping_years = set(dict_1.keys()) & set(dict_2.keys())
        
    correlations = []

    for year in list(overlapping_years):

        corr = np.corrcoef(dict_1[year], dict_2[year])[0,1]

        correlations.append(corr)

    avg_corr = np.mean(correlations)

    print(avg_corr)

    return avg_corr



