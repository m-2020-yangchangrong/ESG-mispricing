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

# 最新版二维分组代码
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


# 在这修改alpha保留几位小数
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


# Low-High
def __calc_longshort_return__(df, decile, group_name, bottom, top):
    diff_r = pd.merge(df.loc[(df[decile] == bottom)], df.loc[(df[decile] == top)],
                      how='inner', on=group_name, suffixes=['_bottom', '_top'])

    diff_r['weight_r'] = diff_r['weight_r_bottom'] - diff_r['weight_r_top']
    diff_r['equal_r'] = diff_r['equal_r_bottom'] - diff_r['equal_r_top']
    diff_r[decile] = 'bottom-top_' + decile
    return diff_r

def form_portfolio_2D(data, factor_data, decile_name, group_name, r_name, weight_name):
    """
    二维分组
    :param data: 输入数据
    :param declie_names: 分位数名称，按什么分组就写什么
    :param group_name: 分位数分组键，如日，月，年
    :param r_name: 收益率（不减rf的收益率）
    :param weight_name: 加权系数 （按什么加权，比如市值加权）
    :return:
    """
    # 二维，单变量
    decile_name = [name + '_decile' for name in decile_name]

    # 按groupby_columns分组求加权收益/平均收益,按时间+分组decile划分投资组合
    groupby_columns = decile_name + group_name
    weight_return = data.groupby(groupby_columns).apply(__wavg__, r_name, weight_name) \
        .reset_index().rename(columns={0: 'weight_r'})

    equal_return = data.groupby(groupby_columns)[r_name].apply(np.mean) \
        .reset_index().rename(columns={r_name: 'equal_r'})

    # 按merge_keys将weight_return,equal_return合并
    merge_keys = decile_name + group_name
    group_return = pd.merge(weight_return, equal_return, on=merge_keys, how='inner')

    decile_name_one = decile_name[0]
    decile_name_two = decile_name[1]
    decile_name_one_bottom = group_return[decile_name_one].min()
    decile_name_two_bottom = group_return[decile_name_two].min()
    decile_name_one_top = group_return[decile_name_one].max()
    decile_name_two_top = group_return[decile_name_two].max()

    decile_name_two_return = group_return.groupby(decile_name_one) \
        .apply(__calc_longshort_return__, decile_name_two, group_name, decile_name_two_bottom,
               decile_name_two_top).reset_index(drop=False)
    decile_name_one_return = group_return.groupby(decile_name_two) \
        .apply(__calc_longshort_return__, decile_name_one, group_name, decile_name_one_bottom,
               decile_name_one_top).reset_index(drop=False)

    # form portfolio data
    use_cols = decile_name + group_name
    use_cols.append('weight_r')
    use_cols.append('equal_r')
    group_return = group_return.append(decile_name_one_return[use_cols])
    group_return = group_return.append(decile_name_two_return[use_cols])
    # 修改，要减去无风险利率rf
    portfolio_data = pd.merge(group_return, factor_data, on=group_name, how='inner')  # 在group_name 如时间或者省份等
    portfolio_data['weight_r'] = portfolio_data['weight_r'] - portfolio_data['rf']
    portfolio_data['equal_r'] = portfolio_data['equal_r'] - portfolio_data['rf']
    portfolio_data = portfolio_data[decile_name + group_name + ['weight_r', 'equal_r']].copy()
    return portfolio_data


# High-Low
def __calc_longshort_return__1(df, decile, group_name, bottom, top):
    diff_r = pd.merge(df.loc[(df[decile] == bottom)], df.loc[(df[decile] == top)],
                      how='inner', on=group_name, suffixes=['_bottom', '_top'])

    diff_r['weight_r'] = diff_r['weight_r_top'] - diff_r['weight_r_bottom']
    diff_r['equal_r'] = diff_r['equal_r_top'] - diff_r['equal_r_bottom']
    diff_r[decile] = 'top-bottom_' + decile
    return diff_r

def form_portfolio_2D_1(data, factor_data, decile_name, group_name, r_name, weight_name):
    """
    二维分组
    :param data: 输入数据
    :param group_name: 分位数分组键，如日，月，年
    :param declie_names: 分位数名称
    :param r_name: 收益率
    :param weight_name: 加权系数
    :return:
    """
    # 二维，单变量
    decile_name = [name + '_decile' for name in decile_name]

    # 按groupby_columns分组求加权收益/平均收益,按时间+分组decile划分投资组合
    groupby_columns = decile_name + group_name
    weight_return = data.groupby(groupby_columns).apply(__wavg__, r_name, weight_name) \
        .reset_index().rename(columns={0: 'weight_r'})

    equal_return = data.groupby(groupby_columns)[r_name].apply(np.mean) \
        .reset_index().rename(columns={r_name: 'equal_r'})

    # 按merge_keys将weight_return,equal_return合并
    merge_keys = decile_name + group_name
    group_return = pd.merge(weight_return, equal_return, on=merge_keys, how='inner')

    decile_name_one = decile_name[0]
    decile_name_two = decile_name[1]
    decile_name_one_bottom = group_return[decile_name_one].min()
    decile_name_two_bottom = group_return[decile_name_two].min()
    decile_name_one_top = group_return[decile_name_one].max()
    decile_name_two_top = group_return[decile_name_two].max()

    decile_name_two_return = group_return.groupby(decile_name_one) \
        .apply(__calc_longshort_return__1, decile_name_two, group_name, decile_name_two_bottom,
               decile_name_two_top).reset_index(drop=False)
    decile_name_one_return = group_return.groupby(decile_name_two) \
        .apply(__calc_longshort_return__1, decile_name_one, group_name, decile_name_one_bottom,
               decile_name_one_top).reset_index(drop=False)

    # form portfolio data
    use_cols = decile_name + group_name
    use_cols.append('weight_r')
    use_cols.append('equal_r')
    group_return = group_return.append(decile_name_one_return[use_cols])
    group_return = group_return.append(decile_name_two_return[use_cols])
    # 修改，要减去无风险利率rf
    portfolio_data = pd.merge(group_return, factor_data, on=group_name, how='inner')  # 在group_name 如时间或者省份等
    portfolio_data['weight_r'] = portfolio_data['weight_r'] - portfolio_data['rf']
    portfolio_data['equal_r'] = portfolio_data['equal_r'] - portfolio_data['rf']
    portfolio_data = portfolio_data[decile_name + group_name + ['weight_r', 'equal_r']].copy()
    return portfolio_data
