import pandas as pd
import statsmodels.formula.api as smf
from functools import reduce
import numpy as np


def calc_r_squared(data,coefPremium,regmodel):
    """
    自己写的fama macbeth计算r2
    :param data:
    :param coefPremium:
    :param regmodel:
    :return:
    """

    index = list(coefPremium.index.array) #系数项顺序，第一个元素位interpret
    indvars=index[1:]# 不包括常数项
    depvar,_=regmodel.replace(' ', '').strip(" ").split("~") #得到因变量名称
    vars=[depvar]+indvars
    data=data[vars]
    y=np.asarray(data[depvar])
    x=np.asarray(data[indvars])
    params=np.asarray(coefPremium['coef'].iloc[1:].T) #得到除常数项的系数值
    fitted = x @params+coefPremium['coef'].iloc[0] #自变量*系数+常数项
    eps = y - fitted #残差

    residual_ss = float(eps @ eps.T) #残差平方
    e = y - y.mean() #因变量-均值
    total_ss = float(e@e.T) #求平方
    r2 = 1 - residual_ss / total_ss
    return r2

def __fama_macbeth_coef__(rdata, regModel, NWlag):
    """
    自己写的fama macbeth 第一步，横截面回归
    :param rdata:
    :param regModel:
    :param NWlag:
    :return:
    """
    ols = smf.ols(formula=regModel, data=rdata).fit(cov_type='HAC', cov_kwds={'maxlags': NWlag})
    ols_result=ols.params
    ols_result['No. Observations']=ols.nobs
    ols_result['R-squared']=ols.rsquared
    ols_result['Adj.R-squared']=ols.rsquared_adj
    return ols_result

def _format_coef_(_row):
    """
    系数format
    """
    _row['t-stat'] = '(%2.2f)' % _row['t-stat']

    if _row['p-value'] <= 0.01:
        _row['coef'] = '%2.3f***' % _row['coef']

    elif 0.01 < _row['p-value'] <= 0.05:
        _row['coef'] = '%2.3f**' % _row['coef']

    elif 0.05 < _row['p-value'] <= 0.1:
        _row['coef'] = '%2.3f*' % _row['coef']

    else:
        _row['coef'] = '%2.3f' % _row['coef']

    return _row

def fama_macbeth(regdata,formula,ols_result, NWlag,model_name):
    """
    自己写的fama macbeth第二步，横截面系数回归
    :param regdata:
    :param formula:
    :param ols_result:
    :param NWlag:
    :param model_name:
    :return:
    """
    nobs=int(ols_result['No. Observations'].sum())
    R2_mean='{:.1%}'.format(ols_result['R-squared'].mean())
    AdjR2_mean='{:.1%}'.format(ols_result['Adj.R-squared'].mean())
    betas=ols_result.drop(['R-squared','Adj.R-squared','No. Observations'],axis=1)
    coefPremium = pd.DataFrame()
    colPremium = betas
    #Fama 回归第二步，对横截面系数求回归(基本与求均值一致，可做newy-west调整)
    for p in colPremium:
        ols = smf.ols(('%s ~ 1' % p), data=betas).fit(cov_type='HAC',
                                                      cov_kwds={'maxlags': NWlag, 'use_correction': True})
        tmpCoef = pd.DataFrame({'coef': ols.params['Intercept'],
                                't-stat': ols.tvalues['Intercept'],
                                'p-value': ols.pvalues['Intercept']
                                }, index=[p])
        coefPremium = coefPremium.append(tmpCoef)
    #需单独计算R2
    R2=calc_r_squared(regdata,coefPremium,formula)

    coefPremium = coefPremium.apply(_format_coef_, axis=1)
    _output = pd.DataFrame()
    for var in coefPremium.index:
        tmp = pd.DataFrame({model_name: coefPremium.drop('p-value', axis=1).loc[var]}).T
        tmp.rename(columns={'coef': var, 't-stat': var + '_t-stat'}, inplace=True)
        _output = _output.append(tmp.T)
    # 其他回归信息
    _info = pd.DataFrame({model_name: ['%2.3f' % R2,
                                        '%d' % nobs,
                                       R2_mean,
                                       AdjR2_mean]},
                         index=[ 'Adj. R-squared', 'No. Observations','R2_mean','AdjR2_mean'])
    _output = _output.append(_info)
    return _output

def merge_results(fama_results,regressor_order,info_list):
    #1.合并多列结果
    def merg(x, y):
        return x.merge(y, how='outer', right_index=True,
                       left_index=True)
    summ = reduce(merg, fama_results)

    # 一.自变量数据
    varnames = summ.index.tolist()
    params = [x[:-7] for x in varnames if "_t-stat" in x]  # 取得变量名

    # 按order顺序排序
    param_ordered = [x for x in regressor_order if x in params]
    param_unordered = [x for x in params if x not in regressor_order]
    ##包含order的字段和不需要order的字段
    all_params = param_ordered + param_unordered
    # 形成param 和param的t值变量
    def f1(all_params):
        return sum([[x, x + '_t-stat'] for x in all_params], [])

    all_params = f1(all_params) #index 字段名称
    all_params_df=summ.loc[all_params]

   # 二 effect数据
    if set(['EntityEffects','TimeEffects']).issubset(set(summ.index.tolist())):
        effects_info=summ.loc[['EntityEffects','TimeEffects']].values.T
        all_effects = []
        # print(type(effects_info))
        # print(effects_info)
        for effects in effects_info:
            # print(effects)
            effects = [effect for effect in effects if effect is not np.nan  ]
            all_effects.append(effects)

        all_effects_df = pd.DataFrame(data=all_effects).T
        all_effects_df.columns=all_params_df.columns
    else:
        all_effects_df=pd.DataFrame()



    # 三 实际带的info信息字段
    info = [x for x in info_list if x in varnames]
    info_df=summ.loc[info]
    info_df.columns=all_params_df.columns

    # 四 拼接顺序，param， effect，info
    dat = pd.concat([all_params_df, all_effects_df])
    dat = pd.concat([dat, info_df])

    if all_effects_df.empty:
        dat.index = pd.Index(all_params_df.index.tolist()+ info_df.index.tolist())
        summ = dat
    elif len(all_effects_df) == 2:
        dat.index = pd.Index(all_params_df.index.tolist() + ['Efftects'] + [''] + info_df.index.tolist())
        summ = dat
    elif len(all_effects_df) == 1:
        dat.index = pd.Index(all_params_df.index.tolist() + ['Efftects'] + info_df.index.tolist())
        summ = dat
    else:
        dat.index = pd.Index(all_params_df.index.tolist() + ['Efftects'] + info_df.index.tolist())
        summ = dat


    #五 将index中的t-stat 索引置为空
    index=[ "" if '_t-stat' in x else x  for x in summ.index]

    #六 重置索引
    summ.index =  index

    #七 填充nan为空
    summ = summ.fillna('')
    return summ

def _format_reg_(_model, _model_name, _is_control, _smf=0):
    """
    回归系数格式化
    """
    if _smf == 0:
        _tstats = _model.tstats
    else:
        _tstats = _model.tvalues

    _indvars=_tstats.index #确保索引顺序（不索引也不会出错）
    _coefs = _model.params[_indvars]
    _pvals = _model.pvalues[_indvars]


    # collect the output
    _out = pd.DataFrame({'coef': _coefs, 't-stat': _tstats, 'p-value': _pvals})
    _out = _out.apply(_format_coef_, axis=1)
    # collect the output
    _output = pd.DataFrame()
    # 回归变量数据
    for var in _indvars:
        tmp = pd.DataFrame({_model_name: _out.drop('p-value', axis=1).loc[var]}).T
        tmp.rename(columns={'coef': var, 't-stat': var + '_t-stat'}, inplace=True)
        _output = _output.append(tmp.T)

    rsquared_adj = np.nan

    if hasattr(_model, 'rsquared_adj'):
        rsquared_adj = _model.rsquared_adj
    else:
        if hasattr(_model, 'rsquared'):
            rsquared_adj = _model.rsquared

    # Effects 信息
    effects = getattr(_model, "included_effects", [])
    TimeEffects=EntityEffects=np.nan
    if "Time" in effects:
        TimeEffects="Time"
    if "Entity" in effects:
        EntityEffects = "Entity"

    # 其他回归信息
    _info = pd.DataFrame({_model_name: [EntityEffects,
                                        TimeEffects,
                                        _is_control,
                                  '%2.3f' % rsquared_adj,
                                  '%d' % _model.nobs]},
                         index=['EntityEffects','TimeEffects','Other Controls','Adj. R-squared','No. Observations'])
    _output = _output.append(_info)
    return _output