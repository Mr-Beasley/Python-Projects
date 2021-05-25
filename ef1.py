import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from pandas.tseries.offsets import DateOffset
from scipy.optimize import minimize
from matplotlib.pyplot import cm

def get_rr_returns(rrperiod):
    """
    Gets return from a set of NAV series

    """
    ier_main= pd.read_excel("Index Data MF ETF.xlsx",header=0,index_col=0,parse_dates=True)
    ier_main.index=pd.to_datetime(ier_main.index,format='%Y%m%d')
    ier_main.index=ier_main.index.date
    ier_main.columns = ier_main.columns.str.replace("_"," ")
    ier=ier_main.sort_index(axis=0)
    move=int(ier.shape[0]*rrperiod/5)
    n=ier.shape[1]
    names= ier.columns.values.tolist()
    for name in ier.columns:
        ier[name]=pd.to_numeric((ier[name]-ier[name].shift(move))/ier[name].shift(move))
    ier=ier.astype(float)
    rets=ier.dropna()
    rets=rets.astype(float)
    return ((1+rets)**(1/rrperiod)-1)

def get_returns():
    """
    Gets return from a set of NAV series

    """
    ier_main= pd.read_excel("Index Data 2016-20.xlsx",header=0,index_col=0,parse_dates=True)
    ier_main.index=pd.to_datetime(ier_main.index,format='%Y%m%d')
    ier_main.columns = ier_main.columns.str.replace("_"," ")
    ier=ier_main.sort_index(axis=0)
    rets=ier.pct_change().dropna()
    return rets

def RR_returns(r):
    return r.mean()

def annualize_vol(r):
    """
    Annualizes the vol of a set of returns
    
    """
    return r.std()

def portfolio_rets(weights,returns):
    """
    Calculates portfolio returns

    """
    return (weights @ returns)

def portfolio_vol(weights, covmat):
    """
    Calculates portfolio volatility

    """
    return (weights @ covmat @ weights.T)**0.5

def portfolio_vols(weights, covmat):
    """
    Calculates portfolio volatility

    """
    weight=weights.values
    ret=(weight @ covmat @ weight.T)**0.5
    ret.index=weights.index
    
    return ret

def minimize_vol(target_return,er,cov):
    """
    Gives the minimum volatility for a particular target return, return series and covariance matrix

    """
    n=er.shape[0]
    init_guess=np.repeat(1/n,n)
    bounds=((0.0,1.0),)*n
    weights_sum_to_1={
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    return_is_target={
        'type':'eq',
        'args': (er,),
        'fun': lambda weights,er: target_return - portfolio_rets(weights,er)
    }
    weights = minimize(portfolio_vol,
                       init_guess,args=(cov,),
                       method='SLSQP', bounds=bounds, options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target))
    return weights.x

def optimal_weights(n_points,er,cov):
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """
    target_rs=np.linspace(er.min(),er.max(),n_points)
    weights=[minimize_vol(target_return,er,cov) for target_return in target_rs]
    return weights

def efficient_frontier(er,cov, show_cml=False, n_points=250,riskfree_rate=0.06, show_ew=False, show_gmv=False):
    weights=optimal_weights(n_points,er,cov)
    rets=[portfolio_rets(w,er) for w in weights]
    vols= [portfolio_vol(w,cov) for w in weights]
    
    r_msr=0
    vol_msr=0
    r_ew=0
    vol_ew=0
    r_gmv=0
    vol_gmv=0
   
    if show_cml:
        w_msr=msr(er,cov,riskfree_rate)
        r_msr=portfolio_rets(w_msr,er)
        vol_msr=portfolio_vol(w_msr,cov)
        cml_x=[0,vol_msr]
        cml_y=[riskfree_rate, r_msr]
        
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_rets(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_rets(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
    
    ef=pd.DataFrame({
        "Return":rets,
        "Volatility":vols,
        "Return_CML":r_msr,
        "Volatility_CML":vol_msr,
        "Return_EW":r_ew,
        "Volatility_EW":vol_ew,
        "Return_GMV":r_gmv,
        "Volatility_GMV":vol_gmv
    })
    return ef

def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(np.repeat(1, n), cov)

def msr(er,cov,riskfree_rate=0.06):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n=er.shape[0]
    init_guess=np.repeat(1/n,n)
    bounds=((0.0,1.0),)*int(n)
    weights_sum_to_1={
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    def neg_SR(weights,riskfree_rate,er,cov):
        r=portfolio_rets(weights,er)
        v=portfolio_vol(weights,cov)
        return -(r-riskfree_rate)/v
    
    weights = minimize(neg_SR,
                       init_guess,args=(riskfree_rate, er, cov),
                       method='SLSQP', bounds=bounds, options={'disp': False},
                       constraints=(weights_sum_to_1))
    return weights.x

def plot_efficient_frontier(n_points,er,cov,df2,style='.-', legend=False, show_cml=False, riskfree_rate=0.06, show_ew=False, show_gmv=False):
    """
    Plots the efficient frontier
    """
    weights=optimal_weights(n_points,er,cov)
    rets=[portfolio_rets(w,er) for w in weights]
    vols= [portfolio_vol(w,cov) for w in weights]
    ef=pd.DataFrame({
        "Return":rets,
        "Volatility":vols
    })
    
    ax=ef.plot.line(x="Volatility",y="Return",style=style,legend=legend,figsize=(20,10))
    ax.set_title('Efficient Frontier Curve')
    
    if show_cml:
        ax.set_xlim(auto=True)
        w_msr=msr(er,cov,riskfree_rate)
        r_msr=portfolio_rets(w_msr,er)
        vol_msr=portfolio_vol(w_msr,cov)
        cml_x=[0,vol_msr]
        cml_y=[riskfree_rate, r_msr]
        ax.plot(cml_x,cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10,label='CML')
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_rets(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # add EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10,label='EW')
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_rets(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # add EW
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10, label ='GMV')
    
    np1=np.linspace(0,df2.shape[0],num=df2.shape[0],endpoint=False,dtype='int64')
    colors = iter(cm.rainbow(np.linspace(0, 1, len(np1))))
    
    for x in np1:
        y=x+1
        c=next(colors)
        ax.plot(df2.iloc[x:y,1:2], df2.iloc[x:y,0:1], color=c, marker='x', markersize=10,label=df2.index[x])
    
    ax.legend()
    return ax


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def sharpe_ratio(r, riskfree_rate):
    """
    Computes the sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    excess_ret = r - riskfree_rate
    ann_ex_ret = RR_returns(excess_ret)
    ann_vol = annualize_vol(r)
    return ann_ex_ret/ann_vol


import scipy.stats
def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level


def drawdown(rets):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index=pd.DataFrame()
    previous_peaks=pd.DataFrame()
    drawdowns=pd.DataFrame()
    wealth_index=wealth_index.astype(float)
    previous_peaks=previous_peaks.astype(float)
    drawdowns=drawdowns.astype(float)
    for name in rets.columns:
        wealth_index[name+" WI"] = pd.to_numeric(1000*(1+rets[name]).cumprod())
        previous_peaks[name+" PP"] = pd.to_numeric(wealth_index[name+" WI"].cummax())
        drawdowns[name] = pd.to_numeric((wealth_index[name+" WI"] - previous_peaks[name+" PP"])/previous_peaks[name+" PP"])
    wealth_index=wealth_index.astype(float)
    previous_peaks=previous_peaks.astype(float)
    drawdowns=drawdowns.astype(float)
    return pd.concat([wealth_index,previous_peaks,drawdowns],axis=1)


def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def sortino_ratio(r, riskfree_rate=0.06):
    dev=r[(r-riskfree_rate)<0].aggregate("std")
    ret=r.aggregate("mean")-riskfree_rate
    return ret/dev
    

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def summary_stats(r, rets, riskfree_rate=0.06):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    rolling_r = RR_returns(r)
    ann_vol = annualize_vol(r)
    ann_sr = sharpe_ratio(r, riskfree_rate=riskfree_rate)
    s_ratio=sortino_ratio(r, riskfree_rate=riskfree_rate)
    skf=drawdown(rets)
    dd = skf.iloc[:,-int(skf.shape[1]/3):].min()
    skew = skewness(rets)
    kurt = kurtosis(rets)
    cf_var5 = var_gaussian(rets, modified=True)
    hist_cvar5 = cvar_historic(rets)
    rolling_r=rolling_r.rename('Rolling Return',inplace=True)
    ann_vol=ann_vol.rename('Annualised Volatility',inplace=True)
    ann_sr=ann_sr.rename('Sharpe Ratio',inplace=True)
    s_ratio=s_ratio.rename('Sortino Ratio',inplace=True)    
    dd=dd.rename('Max Drawdown',inplace=True)
    skew=skew.rename('Skewness',inplace=True)
    kurt=kurt.rename('Kurtosis',inplace=True)
    cf_var5=cf_var5.rename('Gaussian VaR',inplace=True)
    hist_cvar5=hist_cvar5.rename('Historic VaR',inplace=True)
    return pd.concat([rolling_r,ann_vol,ann_sr,s_ratio,dd,skew,kurt,cf_var5,hist_cvar5],axis=1)
