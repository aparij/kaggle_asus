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
from lifelines.estimation import AalenAdditiveFitter as AAF


def conv2(row):
   return dt.strptime(row['year/month(repair)'],"%Y/%m")


def get_season(d):
    month = int(d.split("/")[1])
    if month in [1,2,12]:
        return 1.0
    elif month in [3,4,5]:
        return 2.0
    elif month in [6,7,8]:
        return 3.0
    elif month in [9,10,11]:
        return 4.0

def get_season_name(d):
    month = int(d.split("/")[1])
    if month in [1,2,12]:
        return "winter"
    elif month in [3,4,5]:
        return 'spring'
    elif month in [6,7,8]:
        return 'summer'
    elif month in [9,10,11]:
        return 'fall'


def find_closest(x,num_list):
    """
     Find closest before and after number X in sorted num_list
    Assuming x is not in num_list
    """
           
    num_list = list(num_list)[1:-1]
    for i in num_list:
        if i>x:
            return (num_list.index(i)-1+1,num_list.index(i)+1) #adding 1 because I remove zero in the beginning


def linear_f(x,slope,intercept):
    return x*slope + intercept



data = pd.read_csv('RepairTrain.csv')
sales = pd.read_csv("SaleTrain.csv")
out = pd.read_csv("Output_TargetID_Mapping.csv")




#for lin regress
lin_grp = data.groupby(['module_category','component_category','year/month(repair)'],as_index=False).agg({'number_repair':np.sum})
models = {}
last_dts=[]
last_months = ["2009-10-01", '2009-11-01','2009-12-01']   # '2009-04-01','2009-05-01',    ['2009-06-01','2009-07-01','2009-08-01','2009-09-01',,
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
            print "sum of last months is zero , no need for forecasting"
        else:    


            cur_s = sales[ (sales['module_category']==m) & (sales['component_category']==p )]
            cur_d = data[ (data['module_category']==m) & (data['component_category']==p )]

            d_grp = cur_d.groupby(['year/month(sale)','year/month(repair)'],as_index=False).agg({'number_repair':np.sum})
            s_grp = cur_s.groupby(['year/month'],as_index=False).agg({'number_sale':np.sum})
            print s_grp
            repairs_sale_month = cur_d.groupby(['year/month(sale)'],as_index=False).agg({'number_repair':np.sum})
            data_events = np.array([])
            extra_var = np.array([])
            extra_vars_df = DataFrame([])
            sales_dict = {}

            for r in d_grp.values:
                sale_date = r[0]
                repair_date = r[1]
                num_repair = r[2]
                if '2004' in sale_date or '2003' in sale_date or sale_date=='2005/01':
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
                data_events = np.append(data_events,np.array([time_to_event]*num_repair))
                extra_var = np.append(extra_var,np.array([int(sale_date.split('/')[1])]*num_repair)) #get_season(sale_date)]*num_repair))

            for k,v in sales_dict.iteritems():
                #investigate why some negative leftovers on certain valid dates , more repairs than sales ???
                print v
                if v>0:
                    data_events = np.append(data_events,np.zeros(v))
                    extra_var = np.append(extra_var,np.array([int(k.split('/')[1])]*v))

            t=[]
            if len(data_events)==0:
                all_data.append([0]*19)
                continue

            data_events[data_events==0] = 60
            C= data_events <60
            aaf = AAF(penalizer=0.5)

            #extra_vars_df["winter"] = (extra_var == 1.0).astype(float)
            #extra_vars_df["spring"] = (extra_var == 2.0).astype(float)
            #extra_vars_df["summer"] = (extra_var == 3.0).astype(float)
            #extra_vars_df["fall"] = (extra_var == 4.0).astype(float)

            extra_vars_df[str(1)] = (extra_var == 1).astype(float)
            extra_vars_df[str(2)] = (extra_var == 2).astype(float)
            extra_vars_df[str(3)] = (extra_var == 3).astype(float)
            extra_vars_df[str(4)] = (extra_var == 4).astype(float)
            extra_vars_df[str(5)] = (extra_var == 5).astype(float)
            extra_vars_df[str(6)] = (extra_var == 6).astype(float)
            extra_vars_df[str(7)] = (extra_var == 7).astype(float)
            extra_vars_df[str(8)] = (extra_var == 8).astype(float)
            extra_vars_df[str(9)] = (extra_var == 9).astype(float)
            extra_vars_df[str(10)] = (extra_var == 10).astype(float)
            extra_vars_df[str(11)] = (extra_var == 11).astype(float)
            extra_vars_df[str(12)] = (extra_var == 12).astype(float)


            tmp_df = extra_vars_df.ix[0:len(extra_vars_df),0:13]
            aaf.fit( data_events[:,None], tmp_df.values, censorship = C, columns=extra_vars_df.columns)
            print "CUMULATIVE HAZARD \n\n", aaf.cumulative_hazards_

            aaf_h = aaf.cumulative_hazards_
            x = np.array(aaf.cumulative_hazards_.index).astype(int)
            new_x = []
            new_y = []

            if len(aaf_h) > 14:
                #plt.figure()
                #plt.plot(x, y_h, 'ko')
                #plt.plot(x, linear_f(x,slope,intercept ), 'r-')

                #plt.plot(x, exp_f(x,*popt ), 'r-', label="Fitted Curve")
                #plt.plot(x, compertz2(x,*c_popt ), 'r-', label="Compertz Fitted Curve")

                #plt.legend()
                #plt.show()

                total_repairs={}

                forecast_month = dt.strptime('2010/01',"%Y/%m") 
                for i in range(1,20):
                    #print "------------------", forecast_month
                    total_repairs_month = 0
                    for r in s_grp.values:
                        sale_month = r[0]
                        season = sale_month.split('/')[1] #get_season_name(sale_month)
                        t_sales = r[1]
                        s_dt = dt.strptime(sale_month,"%Y/%m") 
                        r = relativedelta(forecast_month, s_dt )
                        time_to_event =  r.years*12 + r.months

                        if time_to_event> (len(aaf_h[season])-2):
                           h = aaf_h[season].iloc[len(aaf_h)-2] + aaf_h['baseline'].iloc[len(aaf_h)-2]
                           h_before =  aaf_h[season].iloc[len(aaf_h)-3] + aaf_h['baseline'].iloc[len(aaf_h)-3]
                           this_month =  t_sales*(h-h_before)  
                           #what todo with negative cumulative hazard rate
                           if this_month>0:
                                total_repairs_month += this_month
                        else:

                            if time_to_event not in x:#seen_data_events:
                                (before,after) = find_closest(time_to_event,x)#seen_data_events)
                                #average of before and after the point we looking for example if we have 30,31,36,37 and we need 33 , we get before 31 and after 36
                                h = (aaf_h[season].iloc[before] + aaf_h[season].iloc[after] + aaf_h['baseline'].iloc[before] + aaf_h['baseline'].iloc[before] )/2
                                h_before = aaf_h[season].iloc[before] +  aaf_h['baseline'].iloc[before]

                            else:    
                                #use existing hazard data
                                #print "Existing"
                                h = aaf_h[season][time_to_event] + aaf_h['baseline'][time_to_event]
                                ind = np.where(x==time_to_event)[0][0]
                                h_before = aaf_h[season].iloc[ind-1] +  aaf_h['baseline'].iloc[ind-1]

                            this_month =  t_sales*(h-h_before)  
                            #what todo with negative cumulative hazard rate
                            if this_month>0:
                                total_repairs_month += this_month



                           # if time_to_event not in seen_data_events:
                           #     (before,after) = find_closest(time_to_event,x)
                           #     h = (aff_h[season][before] + aaf_h[season][after])/2
                           #     h_before = aaf_h[season][before]
                           # else:    
                           #     #use existing hazard data
                           #     #print "Existing"
                           #     ind = np.where(x==time_to_event)[0][0]
                           #     h = aaf_y[season][ind]
                           #     h_before = aaf_h[season][ind-1]

                    if total_repairs_month >=0:        
                        t.append(int(total_repairs_month))        
                    else:
                        t.append(0)        
                    forecast_month = forecast_month + relativedelta(months=1)
                    new_x.append(time_to_event)
                    new_y.append(h)

            else:
                t = [0]*19

        all_data.append(t)



i=1
k=1
out_list = []
next_value = 0
for r in out.values:
    if k > 19:
        k=1
    try:
        cur_com = all_data[(i-1)/19]    
        out_list.append([i,cur_com[k-1]])
    except:
        continue
        #import pdb
        #pdb.set_trace()
    i = i+1
    k = k+1

np.savetxt("aaf_survival_%s.csv" % str(dt.now()) ,out_list,delimiter=",",header='id,target',fmt='%d')

