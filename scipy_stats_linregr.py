from datetime import datetime as dt , date, time
import sys
import pandas as pd
from pandas import Series, DataFrame, Panel
import matplotlib.pyplot as plt
import matplotlib as mpl
from dateutil.relativedelta import *
import numpy as np
from scipy import stats



def conv(row):
   return dt.strptime(row['year/month(sale)'],"%Y/%m")

def conv2(row):
   return dt.strptime(row['year/month(repair)'],"%Y/%m")

#def deltas(row):
#   r= relativedelta(dt.strptime(row['repair_dt'],"%Y-%m-%d %H:%M:%S") , dt.strptime(row['sale_dt'],"%Y-%m-%d %H:%M:%S"))
#   return r.years*12 + r.months


def deltas(row):
   r= relativedelta(row['repair_dt'], row['sale_dt'] )
   return r.years*12 + r.months


data = pd.read_csv('RepairTrain.csv')
sales = pd.read_csv("SaleTrain.csv")
out = pd.read_csv("Output_TargetID_Mapping.csv")



d_grp = data.groupby(['module_category','component_category','year/month(repair)'],as_index=False).agg({'number_repair':np.sum})

i=1
out_list = []
models = {}
last_dts=[]
last_months = ['2009-06-01','2009-07-01','2009-08-01','2009-09-01',"2009-10-01",'2009-11-01','2009-12-01']   # '2009-04-01','2009-05-01',                       
for m in last_months:
    last_dts.append(dt.strptime(m,'%Y-%m-%d'))
k=1
for r in out.values:
    if k > 19:
        k=1
    m = r[0]
    p = r[1]
    if m+p in models:
       slope = models[m+p]['slope']
       intercept = models[m+p]['intercept']
    else:
        t = d_grp[ (d_grp['module_category']==m) & (d_grp['component_category']==p )]
        t['repair_dt'] = t.apply(conv2,axis=1)
        t = t.sort(['repair_dt'],ascending=True) 
        j=0
        x=np.arange(1,len(last_months)+1)
        y=np.zeros(len(last_months))
        print t[-len(last_months)::].values
        for r in t[-len(last_months)::].values:
            print r[4],"------------==\n"
            if r[4] in last_dts:
               y[last_dts.index(r[4])] = r[3]
            #else:
            #   y = np.append(y,0)
            j = j + 1     
            #x = np.append(x,j)    
        print x,y
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        if np.isnan(slope):
            slope = 0
        if np.isnan(intercept):
            intercept = 0
        print slope,intercept    
        models[m+p] = {'slope':slope,'intercept':intercept}                                
    val = int(slope*(k + len(last_months) ) + intercept )

    if len(out_list)>2:
       if out_list[-1][1]>=out_list[-2][1]:
          val = int(val/4 )  + val
       else:
          val = val - int(val/4 )  

    print k, len(last_months)  , val
    if val<0:
        val=0
    k = k+1
    out_list.append([i,val])
    i = i+1

np.savetxt("jumping_lin_regress.csv",out_list,delimiter=",",header='id,target',fmt='%d')
