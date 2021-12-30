# -*- coding: utf-8 -*-
from __future__ import division  ###小数除法
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import datetime
import os
from linearmodels.panel import PanelOLS, FamaMacBeth
from linearmodels.panel import compare
from scipy.stats.mstats import winsorize
from linearmodels import FamaMacBeth
from linearmodels import BetweenOLS
from linearmodels import PanelOLS

# environment
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


def alpha(portfolio_data, regModel, NWlag=1):
    ols = smf.ols(formula=regModel, data=portfolio_data) \
        .fit(cov_type='HAC', cov_kwds={'maxlags': NWlag})
    output = pd.DataFrame({'c': [ols.params['Intercept']],
                           't-stat.': [ols.tvalues['Intercept']],
                           'p': [ols.pvalues['Intercept']],
                           'Num.Obs.': [ols.nobs],
                           'R-squared': [ols.rsquared_adj]})
    output = output.apply(lambda x: __significant__(x, 'c', 'p'), axis=1)
    output['Coef.'] = output['c*']
    return output[['Coef.', 't-stat.', 'Num.Obs.', 'R-squared']]

    # return output[['Coef.']]

def __significant__(row, coef, pval):
    if row[pval] < 0.01:
        row[coef + '*'] = str(round(row[coef], 4)) + '***'
    elif (row[pval] >= 0.01) & (row[pval] < 0.05):
        row[coef + '*'] = str(round(row[coef], 4)) + '**'
    elif (row[pval] >= 0.05) & (row[pval] < 0.1):
        row[coef + '*'] = str(round(row[coef], 4)) + '*'
    else:
        row[coef + '*'] = str(round(row[coef], 4))
    return row

def __wavg__(group, r_name, weight_name):
    d = group[r_name]
    w = group[weight_name]

    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


# High-Low
def form_portfolio_1D(data, factor_data, decile_name, date_name, r_name, weight_name):
    """
    :param data: 输入数据
    :param date_name: 分位数分组键，如日，月，年
    :param decile_name: 分位数名称
    :param r_name: 收益率
    :param weight_name: 加权系数
    :return:
    """
    # 一维，单变量
    decile_name = [name + "_decile" for name in decile_name]

    # 1. 按时间和分组指标形成投资组合的加权收益/平均收益
    groupby_columns = decile_name + date_name
    weight_return = data.groupby(groupby_columns).apply(__wavg__, r_name, weight_name) \
        .reset_index().rename(columns={0: 'weight_r'})

    equal_return = data.groupby(groupby_columns)[r_name].mean() \
        .reset_index().rename(columns={r_name: 'equal_r'})

    # 2. 按时间和分组指标将weight_return,equal_return合并
    merge_keys = decile_name + date_name
    portfolio_return = pd.merge(weight_return, equal_return, on=merge_keys, how='inner')

    bottom = portfolio_return[decile_name].min().squeeze()
    top = portfolio_return[decile_name].max().squeeze()

    # 3. 计算bottom组和top组的收益差值

    diff_return = pd.merge(portfolio_return.loc[(portfolio_return[decile_name].squeeze() == bottom)],
                           portfolio_return.loc[(portfolio_return[decile_name].squeeze() == top)],
                           how='inner', on=date_name, suffixes=['_bottom', '_top'])

    diff_return['weight_r'] = diff_return['weight_r' + '_top'] - diff_return['weight_r' + '_bottom']
    diff_return['equal_r'] = diff_return['equal_r' + '_top'] - diff_return['equal_r' + '_bottom']
    diff_return[decile_name[0]] = 'top_bottom_' + decile_name[0]  # long short组命名  High-Low

    # 4. 计算正常分组的超额收益
    portfolio_return = pd.merge(portfolio_return, factor_data, on=date_name, how='inner')

    portfolio_return['weight_r'] = portfolio_return['weight_r'] - portfolio_return['rf']
    portfolio_return['equal_r'] = portfolio_return['equal_r'] - portfolio_return['rf']

    # 5. form portfolio data
    use_cols = decile_name + date_name
    use_cols.append('weight_r')
    use_cols.append('equal_r')
    portfolio_return = portfolio_return[use_cols].copy()
    portfolio_return = portfolio_return.append(diff_return[use_cols])
    portfolio_return = portfolio_return.sort_values(date_name + decile_name).reset_index(drop=True)
    return portfolio_return


# Low-High
def form_portfolio_1D_1(data, factor_data, decile_name, date_name, r_name, weight_name):
    """
    :param data: 输入数据
    :param date_name: 分位数分组键，如日，月，年
    :param decile_name: 分位数名称
    :param r_name: 收益率
    :param weight_name: 加权系数
    :return:
    """
    # 一维，单变量
    decile_name = [name + "_decile" for name in decile_name]

    # 1. 按时间和分组指标形成投资组合的加权收益/平均收益
    groupby_columns = decile_name + date_name
    weight_return = data.groupby(groupby_columns).apply(__wavg__, r_name, weight_name) \
        .reset_index().rename(columns={0: 'weight_r'})

    equal_return = data.groupby(groupby_columns)[r_name].mean() \
        .reset_index().rename(columns={r_name: 'equal_r'})

    # 2. 按时间和分组指标将weight_return,equal_return合并
    merge_keys = decile_name + date_name
    portfolio_return = pd.merge(weight_return, equal_return, on=merge_keys, how='inner')

    bottom = portfolio_return[decile_name].min().squeeze()
    top = portfolio_return[decile_name].max().squeeze()

    # 3. 计算bottom组和top组的收益差值

    diff_return = pd.merge(portfolio_return.loc[(portfolio_return[decile_name].squeeze() == bottom)],
                           portfolio_return.loc[(portfolio_return[decile_name].squeeze() == top)],
                           how='inner', on=date_name, suffixes=['_bottom', '_top'])

    diff_return['weight_r'] = diff_return['weight_r' + '_bottom'] - diff_return['weight_r' + '_top']
    diff_return['equal_r'] = diff_return['weight_r' + '_bottom'] - diff_return['weight_r' + '_top']
    diff_return[decile_name[0]] = 'bottom_top_' + decile_name[0]  # long short组命名

    # 4. 计算正常分组的超额收益
    portfolio_return = pd.merge(portfolio_return, factor_data, on=date_name, how='inner')

    portfolio_return['weight_r'] = portfolio_return['weight_r'] - portfolio_return['rf']
    portfolio_return['equal_r'] = portfolio_return['equal_r'] - portfolio_return['rf']

    # 5. form portfolio data
    use_cols = decile_name + date_name
    use_cols.append('weight_r')
    use_cols.append('equal_r')
    portfolio_return = portfolio_return[use_cols].copy()
    portfolio_return = portfolio_return.append(diff_return[use_cols])
    portfolio_return = portfolio_return.sort_values(date_name + decile_name).reset_index(drop=True)
    return portfolio_return
