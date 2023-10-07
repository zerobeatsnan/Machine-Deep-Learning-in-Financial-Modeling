import numpy as np
import pandas as pd
from copy import copy
from tabulate import tabulate
import statsmodels.api as sm

class backtest:
    def __init__(self, prediction, ret, weight, characteristics = None, n_group = 10, freq = 'm'):    
        self.prediction = copy(prediction)
        self.ret = copy(ret)
        self.date_list = list(self.prediction.index)
        self.weight = copy(weight)

        self.prediction[self.ret == 0] = 0
        self.prediction[self.prediction == 0] = np.nan
    
        self.ret = self.ret.loc[self.prediction.index]
        self.weight = self.weight.loc[self.prediction.index]
        self.n_group = n_group
        self.freq = freq

        if characteristics is not None:
            self.Nt = ret[ret != 0].count(axis = 1)
            self.xt = pd.DataFrame(np.nan, index = self.ret.index, columns = characteristics[self.date_list[0]].columns)
            self.xt_pred = pd.DataFrame(np.nan, index = self.ret.index, columns = characteristics[self.date_list[0]].columns)
            for t in self.date_list:
                nt = self.Nt.loc[t]
                self.xt.loc[t] = characteristics[t].T.dot(self.ret.loc[t, characteristics[t].index]) / nt
                self.xt_pred.loc[t] = characteristics[t].T.dot(self.prediction.loc[t, characteristics[t].index].fillna(0)) / nt   
        else:
            self.xt = None
    
    # calculating r2
    def calculating_R2(self):
        self.R2 = 1 - ((self.ret - self.prediction)**2).sum(skipna = True).sum(skipna = True) / (self.ret**2).sum(skipna = True).sum(skipna = True)
        if self.xt is not None:
            self.R2x = 1 - ((self.xt - self.xt_pred)**2).sum(skipna = True).sum(skipna = True) / (self.xt**2).sum(skipna = True).sum(skipna = True)

    def calculating_pricing_error(self):
        if self.xt is not None:
            self.xt_alpha = (self.xt - self.xt_pred).mean(skipna = True)
            self.xt_mean = self.xt.mean(skipna = True)
            self.error_abs = self.xt_alpha.abs().sum() / self.xt_mean.abs().sum()
            self.error_square = (self.xt_alpha**2).sum() / (self.xt_mean**2).sum()

    # calculating ic
    def calculating_ic(self):
        self.ic_series = self.prediction.corrwith(self.ret, axis=1)
        self.ic_series.name = 'ic'
        self.IC = self.ic_series.mean()
        
        # calculating icir
        ic_dt = pd.DataFrame(self.ic_series)
        ic_dt['const'] = 1
        tsmod = sm.OLS(endog = ic_dt['ic'], exog = ic_dt['const'], missing = 'drop').fit()
        self.ICIR = tsmod.tvalues.values[0]

        if self.xt is not None:
            self.ic_series_x = self.xt_pred.corrwith(self.xt, axis=1)
            self.ic_series_x.name = 'ic'
            self.ICx = self.ic_series_x.mean()
            
            # calculating icir
            ic_dt = pd.DataFrame(self.ic_series_x)
            ic_dt['const'] = 1
            tsmod = sm.OLS(endog = ic_dt['ic'], exog = ic_dt['const'], missing = 'drop').fit()
            self.ICIRx = tsmod.tvalues.values[0]

    # portfolio analysis
    def portfolio_analysis(self):
        pre_rank = self.prediction.rank(axis = 1).divide(self.prediction.count(axis = 1), axis=0)
        percentile_interval = 1 / self.n_group
    
        i = 0
        available_stock = pre_rank.where(pre_rank >= i*percentile_interval).where(pre_rank <= (i+1)*percentile_interval)
        available_stock[(1 - available_stock.isna()).astype(bool)] = 1
        available_stock = available_stock.fillna(0)
        
        ret_ew = (available_stock * self.ret).sum(axis = 1) / available_stock.sum(axis = 1)
        ret_ew.name = i + 1
        if self.weight is not None:
            ret_vw = (available_stock * self.ret * self.weight).sum(axis = 1) / (available_stock * self.weight).sum(axis = 1)
            ret_vw.name = i + 1

        for i in range(1, self.n_group):
            available_stock = pre_rank.where(pre_rank >= i*percentile_interval).where(pre_rank <= (i+1)*percentile_interval)
            available_stock[(1 - available_stock.isna()).astype(bool)] = 1
            available_stock = available_stock.fillna(0)
            
            ret_ew_ = (available_stock * self.ret).sum(axis = 1, skipna = True) / available_stock.sum(axis = 1, skipna = True)
            ret_ew_.name = i + 1
            ret_ew = pd.concat([ret_ew, ret_ew_], axis=1)
            
            if self.weight is not None:
                ret_vw_ = (available_stock * self.ret * self.weight).sum(axis = 1, skipna = True) / (available_stock * self.weight).sum(axis = 1, skipna = True)
                ret_vw_.name = i + 1
                ret_vw = pd.concat([ret_vw, ret_vw_], axis=1)

        def calc_NW_tvalue(factor_value):
            tsmod = sm.OLS(endog = factor_value, exog = np.ones(factor_value.shape[0]), hasconst = False)\
                .fit(cov_type = 'HAC', cov_kwds = {'maxlags': 6})
            return tsmod.tvalues.values[0]

        if self.freq == 'a':
            sr_multi = 1
        elif self.freq == 'm':
            sr_multi = np.sqrt(12)
        elif self.freq == 'd':
            sr_multi = np.sqrt(250)

        def calculate_maxdrawdown(ri):
            drawdown = []
            for t in range(ri.shape[0]):
                drawdown.append(ri.iloc[t] - ri.iloc[t:].min())
            return max(drawdown)
        
        ret_ew['diff'] = ret_ew.iloc[:, self.n_group - 1] - ret_ew.iloc[:, 0]
        self.summary_ret_ew = pd.DataFrame(np.nan, index = ret_ew.columns, columns = ['mean(%)', 't value', 'SR', 'max loss(%)', 'max drawdown(%)'])
        self.summary_ret_ew['mean(%)'] = 100 * ret_ew.mean()
        self.summary_ret_ew['SR'] = sr_multi * ret_ew.mean() / ret_ew.std()
        self.summary_ret_ew['max loss(%)'] = - ret_ew.min()
        self.summary_ret_ew['max drawdown(%)'] = [calculate_maxdrawdown(ret_ew[i]) for i in ret_ew.columns]
        
        for i in self.summary_ret_ew.index:
            self.summary_ret_ew.loc[i, 't value'] = calc_NW_tvalue(ret_ew[i])
        
        if self.weight is not None:
            ret_vw['diff'] = ret_vw.iloc[:, self.n_group - 1] - ret_vw.iloc[:, 0]
            self.summary_ret_vw = pd.DataFrame(np.nan, index = ret_vw.columns, columns = ['mean(%)', 't value', 'SR', 'max loss(%)', 'max drawdown(%)'])
            self.summary_ret_vw['mean(%)'] = 100 * ret_vw.mean()
            self.summary_ret_vw['SR'] = sr_multi * ret_vw.mean() / ret_vw.std()
            self.summary_ret_vw['max loss(%)'] = - ret_vw.min()
            self.summary_ret_vw['max drawdown(%)'] = [calculate_maxdrawdown(ret_vw[i]) for i in ret_ew.columns]
            
            for i in self.summary_ret_vw.index:
                self.summary_ret_vw.loc[i, 't value'] = calc_NW_tvalue(ret_vw[i])

        return ret_ew

    def reporting(self):
        if self.xt is None:
            summary_table = tabulate(
                [
                    ['', 'R2(%)', 'IC(%)', 'ICIR'],
                    ['rt' , round(100 * self.R2, 2), round(100 * self.IC, 2), round(self.ICIR, 2)]
                ], tablefmt="grid"
            )
        else:
            summary_table = tabulate(
                [
                    ['', 'R2(%)', 'IC(%)', 'ICIR', 'A(|α|)/A(|μ|)', 'A(α^2)/A(μ^2)'],
                    ['rt', round(100 * self.R2, 2), round(100 * self.IC, 2), round(self.ICIR, 2), '', ''],
                    ['xt', round(100 * self.R2x, 2), round(100 * self.ICx, 2), round(self.ICIRx, 2), round(self.error_abs, 2), round(self.error_square, 2)]
                ], tablefmt="grid"
            )
        summary_table_ret_ew = tabulate(self.summary_ret_ew, headers = self.summary_ret_ew.columns)
        if self.weight is not None:
            summary_table_ret_vw = tabulate(self.summary_ret_vw, headers = self.summary_ret_vw.columns)
        
        # print the report
        if self.weight is not None:
            print(
                tabulate([['A brief summary of the performance']]) + '\n'
                + summary_table + '\n'
                + tabulate([['Results for portfolio analysis by equal weight']]) + '\n'
                + summary_table_ret_ew + '\n'
                + tabulate([['Results for portfolio analysis by equal weight']]) + '\n'
                + summary_table_ret_vw
            )
        else:
            print(
                tabulate([['A Brief summary of the performance']]) + '\n'
                + summary_table + '\n'
                + tabulate([['Results for portfolio analysis by equal weight']]) + '\n'
                + summary_table_ret_ew
            )

    def testing(self):
        self.calculating_R2()
        self.calculating_ic()
        self.portfolio_analysis()
        self.calculating_pricing_error()
        
        self.reporting()

