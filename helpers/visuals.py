import os 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm, patches
import statsmodels.api as sm
import numpy as np

def add_cabinet_periods(ax,fn='helpers/cabinets.csv',min_time=1946,max_time=1967,color='salmon',alpha=.5,linestyle='--',text=True):
  cbp = pd.read_csv(fn,sep='\t')
  cbp['startdate'] = pd.to_datetime(cbp.startdate,format='%d-%M-%Y')
  cbp['enddate'] = pd.to_datetime(cbp.enddate,format='%d-%M-%Y')
  ymin,ymax = ax.get_ylim()

  for i,r in cbp.iterrows():
      sd = r['startdate']
      if sd.year >= min_time and sd.year <= max_time and r['cabinet'] != 'Van Agt III':
          ax.vlines(x=sd,ymin=ymin,ymax=ymax,color=color,linestyle=linestyle,alpha=alpha,linewidth=1)
          if text == True:
            ax.text(x=sd,y=ymin,s=r['cabinet'],rotation=90)

def add_year_lines(ax,month="01",min_time=1946,max_time=1967,color='gray',alpha=.5,linestyle='--'):
    ymin,ymax = ax.get_ylim()
    ax.vlines(x=[f'{y}-{month}-01' for y in range(min_time,max_time)], ymin=ymin, ymax=ymin, color=color, ls=linestyle,alpha=alpha, lw=.75)

def plot_trend(x, y, ax, color='blue', alpha=0.2,ls='-'):
    # Convert pandas datetime objects to numerical values (timestamps)
    if isinstance(x[0], pd.Timestamp):
        x = x.map(pd.Timestamp.timestamp)
    
    x = np.array(x)
    y = np.array(y)
    
    # Fit regression model
    X = sm.add_constant(x)  # Adds a constant term to the predictor
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    
    # Get prediction intervals
    pred = model.get_prediction(X)
    pred_summary = pred.summary_frame(alpha=alpha)
    
    # Convert x back to original datetime format for plotting
    if isinstance(x[0], float):
        x = pd.to_datetime(x, unit='s')
    
    # Plot regression line
    ax.plot(x, predictions, color=color,alpha=alpha,linestyle=ls)