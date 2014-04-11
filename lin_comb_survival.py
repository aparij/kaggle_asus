from datetime import datetime as dt , date, time, timedelta
import sys
import pandas as pd
from pandas import Series, DataFrame, Panel
import matplotlib.pyplot as plt
import matplotlib as mpl
from dateutil.relativedelta import *
import numpy as np
from scipy import stats
from statsmodels.tsa.base.datetools import dates_from_str
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from scipy.optimize import curve_fit



data = pd.read_csv('RepairTrain.csv')
sales = pd.read_csv("SaleTrain.csv")
out = pd.read_csv("Output_TargetID_Mapping.csv")



def find_closest(x,num_list):
    """
     Find closest before and after number X in sorted num_list
    Assuming x is not in num_list
    """
    
    num_list = list(num_list)
    for i in num_list:
        if i>x:
            return (num_list.index(i)-1,num_list.index(i))


def conv2(row):
   return dt.strptime(row['year/month(repair)'],"%Y/%m")

def linear_f(x,slope,intercept):
   return x*slope + intercept
            
#for lin regress
lin_grp = data.groupby(['module_category','component_category','year/month(repair)'],as_index=False).agg({'number_repair':np.sum})
models = {}
last_dts=[]
last_months = ["2009-10-01", '2009-11-01','2009-12-01'] 
for m in last_months:
    last_dts.append(dt.strptime(m,'%Y-%m-%d'))
####################

       


needed_out_vars={}
for r in out.values:
    needed_out_vars.setdefault(r[0],[]).append(r[1])

modules = [ 'M'+ str(i) for i in range(1,10)]
components = []
for i in range(1,10):
    components.append('P0' + str(i))
for i in range(10,32):
    components.append('P' + str(i))
all_data = []

count_missing = 0 
total_count = 0
for m in modules:
    for p in components:
        if p not in needed_out_vars[m]:
            continue
        print "--------------------   ", m ,p,"  ----------------------------------------------\n\n" 

        l = lin_grp[ (lin_grp['module_category']==m) & (lin_grp['component_category']==p )]
        l['repair_dt'] = l.apply(conv2,axis=1)
        l = l.sort(['repair_dt'],ascending=True) 

        sum_last = 0
        for r in l[-len(last_months)::].values:
            if r[4] in last_dts:
                  sum_last += r[3]
        if sum_last <2:
            t = [0]*19
            print "sum of repairs for the last months is zero , no need for forecasting"
        else:    

            cur_s = sales[ (sales['module_category']==m) & (sales['component_category']==p )]
            cur_d = data[ (data['module_category']==m) & (data['component_category']==p )]

            d_grp = cur_d.groupby(['year/month(sale)','year/month(repair)'],as_index=False).agg({'number_repair':np.sum})
            s_grp = cur_s.groupby(['year/month'],as_index=False).agg({'number_sale':np.sum})
            repairs_sale_month = cur_d.groupby(['year/month(sale)'],as_index=False).agg({'number_repair':np.sum})
            data_events = np.array([])
            sales_dict = {}
            seen_data_events = set()
            for r in d_grp.values:
                sale_date = r[0]
                repair_date = r[1]
                num_repair = r[2]
                if '2004' in sale_date or '2003' in sale_date :#or sale_date=='2005/01':
                    continue
                if sale_date not in sales_dict:
                    total_sales = s_grp[s_grp['year/month']==sale_date]['number_sale']
                    #for erroneus entries in repairs , with non existing sales months
                    if len(total_sales)==0:
                        continue
                    sales_dict.setdefault(sale_date,total_sales.values[0])

                #surviving components to be censored later
                sales_dict[sale_date] = sales_dict[sale_date] - num_repair
                sale_dt = dt.strptime(sale_date,"%Y/%m")
                repair_dt = dt.strptime(repair_date,"%Y/%m")
                r = relativedelta(repair_dt, sale_dt )
                time_to_event =  r.years*12 + r.months
                seen_data_events.add(time_to_event)

                data_events = np.append(data_events,np.array([time_to_event]*num_repair))
            for v in sales_dict.values():
                #investigate why some negative leftovers on certain valid dates , more repairs than sales ???
                if v>0:
                    data_events = np.append(data_events,np.zeros(v))

            t=[]
            if len(data_events)==0:
                all_data.append([0]*19)
                continue

            data_events[data_events==0] = 70
            C= data_events <70
            naf = NelsonAalenFitter()
            naf.fit(data_events, event_observed=C )
            y_h =  np.array(naf.cumulative_hazard_).reshape(len(naf.cumulative_hazard_))
            x= np.array(naf.cumulative_hazard_.index).astype(int)

            seen_data_events.add(0)
            seen_data_events.add(70)

            if len(y_h) > 14:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x[len(x)-5:len(x)-1],y_h[len(y_h)-5:len(y_h)-1])
                
                #plt.figure()
                #plt.plot(x, y_h, 'ko')
                #plt.plot(x, linear_f(x,slope,intercept ), 'r-')

                #plt.legend()
                #plt.show()

                total_repairs={}

                forecast_month = dt.strptime('2010/01',"%Y/%m") 
                for i in range(1,20):
                    total_repairs_month = 0
                    for r in s_grp.values:
                        total_count += 1 
                        sale_month = r[0]
                        t_sales = r[1]
                        s_dt = dt.strptime(sale_month,"%Y/%m") 
                        r = relativedelta(forecast_month, s_dt )
                        time_to_event =  r.years*12 + r.months
                        if time_to_event> (len(y_h)-2):
                            #use forecasting when not enough points in NAF cumulative hazards, usually it's the tail end with >45 months since sale
                            h = linear_f(time_to_event,slope,intercept )
                            h_before = linear_f(time_to_event-1,slope,intercept )
                            #slowly decrease the forecasting of linear regre
                            distance = time_to_event - (len(y_h)-2)
                            ratio = (20.0-distance) /20.0

                            if ratio == 0:
                                ratio = 0.03
                            if ratio <0:
                                ratio =  (ratio + 1.0)/30.0
                            if ratio > 1:
                                ratio = 0
                            #manual increase for the summer months....    
                            if forecast_month.month in [6,7,8]:
                                ratio = ratio*1.4
                            total_repairs_month += t_sales*(h-h_before)*1.7*ratio
                            
                        else:
                            if time_to_event not in seen_data_events:
                                (before,after) = find_closest(time_to_event,seen_data_events)
                                #average of before and after the point we looking
                                h = (y_h[before] + y_h[after])/2.0
                                h_before = y_h[before]
                                count_missing += 1 

                            else:    
                                #use existing hazard data
                                h = y_h[time_to_event]
                                h_before = y_h[time_to_event-1]
                            ratio  = 1.0    
                            #summer months is the most useful parameter,  the rest of seasons don't work . increase the hazard a bit
                            if forecast_month.month in [6,7,8]:
                                ratio = ratio*1.4

                            total_repairs_month += t_sales*(h-h_before)*ratio  

                    t.append(int(total_repairs_month))        
                    forecast_month = forecast_month + relativedelta(months=1)
            else:
                #not enough data, repairs etc... just assume everything is zero
                t = [0]*19
    
        all_data.append(t)


#output data
i=1
k=1
out_list = []
next_value = 0
for r in out.values:
    if k > 19:
        k=1
    cur_com = all_data[(i-1)/19]    
    out_list.append([i,cur_com[k-1]])
    i = i+1
    k = k+1

np.savetxt("lin_comb_survival_%s.csv" % str(dt.now()) ,out_list,delimiter=",",header='id,target',fmt='%d')
