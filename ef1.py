import numpy as np
import pandas as pd
import math
from scipy.optimize import minimize

def get_returns():
    """
    Gets return from a set of NAV series

    """
    ier= pd.read_excel("Ef.xlsx",header=0,index_col=0,parse_dates=True)
    ier.index=pd.to_datetime(ier.index,format='%Y%m%d')
    ier.columns = ier.columns.str.strip()
    rets=ier.pct_change().dropna()
    return rets

def annualised_returns(r,periods_per_year):
    compound=(1+r).prod()
    return compound**(periods_per_year/r.shape[0])-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    
    """
    return r.std()*(periods_per_year**0.5)

def portfolio_rets(weights,returns):
    """
    Calculates portfolio returns

    """
    return (weights.T @ returns)

def portfolio_vol(weights, covmat):
    """
    Calculates portfolio volatility

    """
    return (weights.T @ covmat @ weights)**0.5

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
    target_rs=np.linspace(er.min(),er.max(),n_points)
    weights=[minimize_vol(target_return,er,cov) for target_return in target_rs]
    return weights

def efficient_frontier(n_points,er,cov, show_cml=False, riskfree_rate=0.01, show_ew=False, show_gmv=False):
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
        w_msr=msr(riskfree_rate,er,cov)
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
    return msr(0, np.repeat(1, n), cov)

def msr(riskfree_rate,er,cov):
    n=er.shape[0]
    init_guess=np.repeat(1/n,n)
    bounds=((0.0,1.0),)*n
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

def plot_efficient_frontier(n_points,er,cov,df2,style='.-', legend=False, show_cml=False, riskfree_rate=0.01, show_ew=False, show_gmv=False):
    weights=optimal_weights(n_points,er,cov)
    rets=[portfolio_rets(w,er) for w in weights]
    vols= [portfolio_vol(w,cov) for w in weights]
    ef=pd.DataFrame({
        "Return":rets,
        "Volatility":vols
    })
    ax=ef.plot.line(x="Volatility",y="Return",style=style,legend=legend)
    
    if show_cml:
        ax.set_xlim(left=0)
        w_msr=msr(riskfree_rate,er,cov)
        r_msr=portfolio_rets(w_msr,er)
        vol_msr=portfolio_vol(w_msr,cov)
        cml_x=[0,vol_msr]
        cml_y=[riskfree_rate, r_msr]
        ax.plot(cml_x,cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_rets(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # add EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_rets(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # add EW
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
    
    np1=np.linspace(0,df2.shape[0],num=df2.shape[0],endpoint=False,dtype='int64')
    
    for x in np1:
        ax.plot(df2.iloc[x:x,1:2], df2.iloc[x:x,0:1], color='red', marker='o', markersize=10)
    return ax

