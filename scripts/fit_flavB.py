import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('flavB_combined .csv')
#print(df)


#def y(x,a,b,c):
#def y(x,a,b):
def y(x,a,b,c,d):
    return a*np.exp(b*x) + c*pow(x,d)


# left edge  right edge         x       Mu data         Mu MC      Ele data        Ele MC    Total Data      Total MC
maxfev=10000
for col_data, col_mc in [['Mu data','Mu MC'],['Ele data','Ele MC'],['Total Data','Total MC']]:

    fig, ax = plt.subplots()
    data_uncertainty = np.sqrt(df[col_data])
    potp, pcov = curve_fit(y, df['x'], df[col_data], sigma=data_uncertainty, maxfev=maxfev)
    x = np.linspace(0.0066,.3,300)
    data_color = '#3f90da'
    #fit_text = 'fit: y= %5.3f ^2 + %5.3f * x ^ %5.3f' % tuple(potp)
    fit_text = 'fit: y= %5.3f * exp(%5.3f * x) + %5.3f * x ^ %5.3f' % tuple(potp) # %5.3f ^2 + %5.3f * x ^ %5.3f' % tuple(potp)
    ax.plot(x, y(x, *potp), label=fit_text, color=data_color)
    ax.errorbar(df['x'], df[col_data], yerr=data_uncertainty, xerr=None, color=data_color, ls='none', label=col_data, fmt='o')
    #ax.scatter(df['x'], df[col_data], label=col_data, color=data_color, marker='o')
    print(f'{col_data} : {fit_text}')

    mc_uncertainty = np.sqrt(df[col_mc])
    potp, pcov = curve_fit(y, df['x'], df[col_mc], sigma=mc_uncertainty, maxfev=maxfev)
    x = np.linspace(0.0066,.3,300)
    mc_color = '#ffa90e'
    fit_text = 'fit: y= %5.3f * exp(%5.3f * x) + %5.3f * x ^ %5.3f' % tuple(potp) # %5.3f ^2 + %5.3f * x ^ %5.3f' % tuple(potp)
    ax.plot(x, y(x, *potp), label=fit_text, color=mc_color)
    #ax.scatter(df['x'], df[col_mc], label=col_mc, color=mc_color, marker='^')
    ax.errorbar(df['x'], df[col_mc], yerr=mc_uncertainty, xerr=None, color=mc_color, ls='none', label=col_mc, fmt='^')
    print(f'{col_mc} : {fit_text}')




    ax.legend()
    ax.set_xlabel('flavB')
    ax.set_ylabel('yield')
    fig.savefig(f'fit_flavB_{col_data.replace(" data", "")}.png')
    plt.close('all')
