from datetime import datetime as dt , date, time
import sys
import pandas as pd
from pandas import Series, DataFrame, Panel
import matplotlib.pyplot as plt
import matplotlib as mpl
from dateutil.relativedelta import *
import numpy as np
from scipy import stats
from statsmodels.tsa.base.datetools import dates_from_str
from statsmodels.tsa.api import VAR

data = pd.read_csv('RepairTrain.csv')
sales = pd.read_csv("SaleTrain.csv")
out = pd.read_csv("Output_TargetID_Mapping.csv")

needed_out_vars={}
for r in out.values:
    needed_out_vars.setdefault(r[0],[]).append(r[1])


d_grp = data.groupby(['module_category','component_category','year/month(repair)'],as_index=False).agg({'number_repair':np.sum})
d = dates_from_str(d_grp['year/month(repair)'])
d_grp['d'] = d
d_grp = d_grp.sort(['module_category','component_category','d'],ascending=True)
d_grp.index = pd.DatetimeIndex(d_grp['d'])
del d_grp['year/month(repair)']

modules = [ 'M'+ str(i) for i in range(1,10)]
components = []
for i in range(1,10):
    components.append('P0' + str(i))
for i in range(10,32):
    components.append('P' + str(i))


#2007/03
#2009/12
index_list = ['2006/6','2006/7','2006/8','2006/9','2006/10','2006/11','2006/12','2007/01','2007/02','2007/03','2007/04','2007/05','2007/06','2007/07','2007/08','2007/09','2007/10','2007/11','2007/12','2008/01','2008/02','2008/03','2008/04','2008/05','2008/06','2008/07','2008/08','2008/09','2008/10','2008/11','2008/12','2009/01','2009/02','2009/03','2009/04','2009/05','2009/06','2009/07','2009/08','2009/09','2009/10','2009/11','2009/12']
#dates_index = [dates_from_str(i) for i in index_list]
dates_index = dates_from_str(index_list)
no_forecast = {}
forecast = {}
forecast_cols = {}
original_df = {}
for m in modules:
    print "--------------------   ", m ,"  ----------------------------------------------" 
    df = pd.DataFrame(index=dates_index)
    for p in components:
        if p not in needed_out_vars[m]:
            continue
        data_m_c = d_grp[(d_grp['module_category']==m) & (d_grp['component_category']==p) ]
        s = pd.Series(data_m_c['number_repair'] , index=data_m_c['d'] )
        df[p] = s

        #df = pd.DataFrame({'p13':s1,'p12':s2,'p11':s3})
    #print df    
    df = df.fillna(0)
    original_df[m] = df
    stat_df = df.diff().dropna()

    
    #get rid of columns that are zeros at the end , we just assume they will continue to be zeros
    for col_name in stat_df.columns.values:
        if stat_df[col_name][-1] == 0 and  stat_df[col_name][-2] == 0:# and stat_df[col_name][-3] == 0:
            print col_name
            del stat_df[col_name]
            no_forecast.setdefault(m,[]).append(col_name)
    #print stat_df
    forecast_cols[m] = stat_df.columns.values
    #new_df = stat_df[['P17','P15','P16']]
    model = VAR(stat_df)
    maxlags = 3
    try:
        results = model.fit(maxlags, ic='aic', verbose=True)
    except Exception,exc:
        maxlags = 1         
        results = model.fit(maxlags, ic='aic', verbose=True)
    
    #if m == 'M2':
    #   import pdb
    #   pdb.set_trace()
    #import pdb
    #pdb.set_trace()
#    results = model.fit(4)
    #print results.summary()
    lag_order = results.k_ar
    #print "lag_order\n" ,lag_order
    #print "stat_df.values[-log_order] ---\n", stat_df.values[-lag_order:]
    #print "----------------\n"
    #import pdb
    #pdb.set_trace()
    print stat_df.values[-lag_order:]
    print "---------FORECAST------\n" 

    forecast[m] = results.forecast(stat_df.values[-lag_order:], 19)
    print r
    print "-------------------------no_forecast----\n",no_forecast.items()


i=1
k=1
out_list = []
next_value = 0
for r in out.values:
    if k > 19:
        k=1
    m = r[0]
    p = r[1]
    print m,p,"--k--",k
    if m=='M2':
        val=0
    elif m in no_forecast and p in no_forecast[m]:
        val=0
    else:
        f = forecast[m]
        orig = original_df[m]
        index_col = np.where(forecast_cols[m]==p)[0][0]
        forecasted_change = int(f[k-1][index_col])
        print "forecasted change\n", forecasted_change
        print "orig\n", orig[p][-1]
        if k==1:
            val = orig[p][-1] + forecasted_change
            
        else:
            val = next_value + forecasted_change
        next_value = val
        if val<0:
            val=0
                
        #import pdb
        #pdb.set_trace()
    #if m=='M2':
    #   import pdb
    #   pdb.set_trace()
    
    out_list.append([i,val])
    i = i+1
    k = k+1

np.savetxt("fixed_var.csv",out_list,delimiter=",",header='id,target',fmt='%d')
