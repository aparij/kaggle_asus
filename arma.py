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
import statsmodels.api as sm



def smooth(x,window_len):
    
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        w = np.hamming(window_len)

        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]

def arma_fit(df,p,steps):
   return sm.tsa.ARMA(df[p], (19,0)).fit(method='css',solver="powell" )

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
#del d_grp['year/month(repair)']

modules = [ 'M'+ str(i) for i in range(1,10)]
components = []
for i in range(1,10):
    components.append('P0' + str(i))
for i in range(10,32):
    components.append('P' + str(i))

index_list = ['2006/6','2006/7','2006/8','2006/9','2006/10','2006/11','2006/12','2007/01','2007/02','2007/03','2007/04','2007/05','2007/06','2007/07','2007/08','2007/09','2007/10','2007/11','2007/12','2008/01','2008/02','2008/03','2008/04','2008/05','2008/06','2008/07','2008/08','2008/09','2008/10','2008/11','2008/12','2009/01','2009/02','2009/03','2009/04','2009/05','2009/06','2009/07','2009/08','2009/09','2009/10','2009/11','2009/12']
new_dates = ['2010/1','2010/2','2010/3','2010/4','2010/5','2010/6','2010/7','2010/8','2010/9','2010/10','2010/11','2010/12','2011/1','2011/2','2011/3','2011/4','2011/5','2011/6','2011/7']
index_list.extend(new_dates)

dates_index = dates_from_str(index_list)

out_list = []
i=1
for m in modules:

    df = pd.DataFrame(index=dates_index)

    for p in components:
        print "--------------------   ", m , p ,"  ----------------------------------------------" 
        if p not in needed_out_vars[m]:
            continue
        data_m_c = d_grp[(d_grp['module_category']==m) & (d_grp['component_category']==p) ]
        s = pd.Series(data_m_c['number_repair'] , index=data_m_c['d'] )
        
        df[p] = s
        df[p] = df[p].fillna(0)
        original_s = df[p]

        df[p] = smooth(df[p], len(df[p]))
        df[p] = df[p].diff()
        df[p] = df.fillna(0)
        df[p][0] = 0
        
        print df[p]
        print df[p][(len(index_list)-len(new_dates)-5):(len(index_list)-len(new_dates))].values
        if np.sum(  df[p][len(index_list)-len(new_dates)-3:len(index_list)-len(new_dates)].values) == 0:
            print "Skipping prediction ..... "
            for k in range(0,19):
                out_list.append([i+k,0])
            i=i+19      
            continue
        
        #for some large calculations (e.g. M2P06) numpy fails , but doesnt matter because prediction is negative in the end    
        steps = 19
        fit_ready=False
        while not fit_ready and steps>0:
            try:    
                arma_mod20 =  arma_fit(df,p,steps)
                fit_ready = True
                arma_mod20 = sm.tsa.ARMA(df[p], (19,0)).fit()
            except Exception , e:
                print e
                print "Fit failed with %s steps , will try with 3 less steps!............." %steps
                steps = steps - 20
                    
        start  = '2010-01-%s' % dt.now().day
        end =  '2011-7-%s'   % dt.now().day
        
        if not fit_ready:
            print "Fit failed with all steps , setting to 0"
            results = [0]*19
        else:

            results_diff = arma_mod20.predict(start=start ,end=end, dynamic= True)    
            last_value = original_s[len(index_list)-1]
            print "Last Valeu --------",last_value
            results = []
            #results = results_diff
            for diff_val in results_diff:
               last_value = last_value + diff_val 
               results.append(last_value)
        for k in range(0,19):
            val = round(results[k])
            if val<0:
                out_list.append([i+k,0])
            else:
                if steps!=19 and k>steps:
                    out_list.append([i+k, 0])
                else:    
                    out_list.append([i+k, val])
                    
        print "RESULS --------\n",results
        print "Last i -------", i
        i = i+19
        print "---i---", i
        print " original \n",df[p][-26:] 

np.savetxt("fixed_arma.csv",out_list,delimiter=",",header='id,target',fmt='%d')
