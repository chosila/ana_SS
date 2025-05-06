# %%
#from IPython.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))

# %%
import os, sys
import numpy as np
from collections import OrderedDict as OD
import math
#import uproot3
import uproot as uproot
import hist
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import enum
import mplhep as hep
from parse import *


from HistogramListForPlottingDataVsMC_TriggerStudy_GGFMode import *

sys.path.insert(1, '../') # to import file from other directory (../ in this case)

from htoaa_Settings import *


## write a way to input the link to the root file from command line and automatically put the resulting plot in the corresponding directory
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('inputFile')

args = parser.parse_args()
print('input file: ', args.inputFile)

sIpFile = args.inputFile
sOpDir_substr = sIpFile.split('/')[-4:-2]


sOpDir = '/afs/cern.ch/work/c/csutanta/HTOAA_CMSSW/BBQQ_calibration/plots/v1/calculate_non_top_mc_sf/'

print('output dir: ', sOpDir)

cmsWorkStatus                  = 'Work in Progress'
era                            = '2018'


if not os.path.exists(sOpDir):
    os.makedirs(sOpDir)

fIpFile = uproot.open(sIpFile)
colors_bkg_list = [
    # ['color', <transperent>, '<fill pattern>']
    ['lightcoral', 0.9, ''],
    ['cyan', 0.9, '' ],
    ['burlywood', 0.9, '' ],
    ['saddlebrown', 0.9, '' ],
    ['slateblue', 0.9, '' ],
    ['lightpink', 0.9, 'xx' ],
    ['darkkhaki', 0.9, '' ],
    ['antiquewhite', 0.9, '//' ],
    ['limegreen', 0.9, '' ],
    ['violet', 0.9, '' ],
    ['firebrick', 0.9, '' ],
    ['darkorchid', 0.9, '' ],
    ['tan', 0.9, '' ],
    ['olive', 0.9, '' ],
    ['purple',  0.9, ''],
]


#histo_name = 'hLeadingFatJetPt'
histo_name = 'flavB_max_jet'
selectionTag = 'sel_0b'  #selectionTags[0]
systematic = 'central'
ExpData_list = ['SingleMuon_Run2018A', 'SingleMuon_Run2018B', 'SingleMuon_Run2018C', 'SingleMuon_Run2018D']
#ExpData_list = ['EGamma_Run2018A', 'EGamma_Run2018B', 'EGamma_Run2018C', 'EGamma_Run2018D']

sData = 'Data ABCD'
histo_name_toUse = '%s_%s' % (histo_name, selectionTag)
hData = None
hBkgTot_values = None
hStack_values_list = np.array([])
sStack_list = []
nBkgTot = 0



hBkg_list = []
sBkg_list = []
hBkg_integral_list = []

## groups for fatJetPt
#groups = [180] + [16]*20 + [32]*6 + [48]*2 + [96]*7 + [40]
# groups = [180] + [16]*20 + [32]*6 + [48]*2 + [96]*6 + [136]
#groups = [180] + [32]*16 + [48]*2 + [96]*6 + [136]
#groups = [180] + [48]*13 + [96]*6 + [120]

## groups for flavB
groups = [4]*50

for dataset in MCBkg_list:
    histo_name_toUse_full = 'evt/%s/%s_%s' % (dataset, histo_name_toUse, systematic)
    h = fIpFile[histo_name_toUse_full].to_hist()

    ## scale histogram by 0.6 if bb/bbq/bbqq
    if ('TTToSemiLeptonic_powheg_bb' in histo_name_toUse_full) or ('TTTo2L2Nu_powheg_bb' in histo_name_toUse_full):
        h=h*0.6


    ## rebin variable bin width by pt
    rebin = hist.rebin(groups=groups)
    h = h[::rebin]

    nTot_ = h.values().sum()
    hBkg_list.append(h)
    sBkg_list.append(dataset)
    hBkg_integral_list.append(nTot_)

    if abs(nTot_ - 0) < 1e-10: continue


idx_hBkg_sortedByIntegral = sorted(range(len(hBkg_integral_list)), key=lambda i: hBkg_integral_list[i], reverse=True)
hStack_list = [ hBkg_list[idx] for idx in idx_hBkg_sortedByIntegral ]
sStack_list = [ sBkg_list[idx] for idx in idx_hBkg_sortedByIntegral ]

## shortening some labels in sStack_list
renamed_sStack_list = []
for n in sStack_list:
    newn = n.replace('Hadronic', 'Had')
    newn = newn.replace('SemiLeptonic', '1L1Nu')
    newn = newn.replace('_Incl', '')
    newn = newn.replace('_powheg', '')
    renamed_sStack_list.append(newn)

sStack_list = renamed_sStack_list
hStack_values_list    = np.array( [ h.values() for h in hStack_list ] )
hStack_edges          = hStack_list[0].axes[0].edges
hBkgTot_values        = np.sum(hStack_values_list, axis=0)

hData = None
for ExpData_component in ExpData_list:
    histo_name_toUse_full = 'evt/%s/%s_%s' % (ExpData_component, histo_name_toUse, systematics_forData)
    h = fIpFile[histo_name_toUse_full].to_hist()
    if hData == None: hData = h
    else:             hData = hData + h


rebin  = hist.rebin(groups=groups)
hData  = hData[::rebin]

xError = (hData.axes[0].edges[1:] - hData.axes[0].edges[0:-1]) / 2


## scale the MC to the data
mc_sf = hData.values().sum()/hBkgTot_values.sum()## scale factor
tot = 0
scaled_hStack_list = []
for h_mc in hStack_list:
    scaled_hStack_list.append(h_mc * mc_sf)
    tot = tot + (h_mc.values().sum())
hStack_list = scaled_hStack_list

## subtract the ttbar and single top from both
hStack_list_tt_st = []
sStack_list_tt_st = []
hStack_list_other = []
sStack_list_other = []
## first split the mc into ttbar+singletop or other
for hname,h_mc in zip(sStack_list, hStack_list) :
    if ('TTTo' in hname) or ('SingleTop' in hname):

        hStack_list_tt_st.append(h_mc)
        sStack_list_tt_st.append(hname)
    else:
        hStack_list_other.append(h_mc)
        sStack_list_other.append(hname)

## substract the ttbar+singletop from the data
hData_values_toUse = hData.values()
for h_MC in hStack_list_tt_st:
    hData_values_toUse = hData_values_toUse - h_MC.values()


hStack_values_list  = np.array( [ h.values() for h in hStack_list_other ] )



## plotting
colors_toUse = [ colors_bkg_list[i][0] for i in range(len(hStack_list_other)) ]
alpha_toUse  = [ colors_bkg_list[i][1] for i in range(len(hStack_list_other)) ]
hatch_toUse  = [ colors_bkg_list[i][2] for i in range(len(hStack_list_other)) ]

fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(8,10), sharex='col', gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
print(hStack_values_list, hData_values_toUse,)
## plotting MC
hep.histplot(
    hStack_values_list,
    bins=hStack_edges,
    ax=ax[0],
    histtype='fill',
    stack=True,
    label=sStack_list_other,
    color=colors_toUse,
    alpha=alpha_toUse,
    hatch=hatch_toUse,
    sort='yield'
)
## plotting data
hep.histplot(
    hData_values_toUse,
    bins=hData.axes[0].edges,
    ax=ax[0],
    #yerr=hData_errors_toUse,
    histtype='errorbar',
    color='black',
    label='%s' % (sData)
)

## plotting ratio
hBkgTot_values = np.sum(hStack_values_list, axis=0)
ratio_values = np.divide(hData_values_toUse, hBkgTot_values, where=hBkgTot_values!=0, out=np.ones(hData.shape))
hep.histplot(
    ratio_values,
    bins=hData.axes[0].edges,
    ax=ax[1],
    #yerr=ratio_error,
    histtype='errorbar',
    color='black',
    label='Data'
)

ax[0].set_ylim(min(hData_values_toUse), max(hData_values_toUse)*1.8)
#ax[0].set_xlim(180, 1500)
ax[0].set_xlim(0,0.3)
ax[0].legend(fontsize=12, ncol=2)
ax[1].set_ylim(-.1, 2.3)
ax[1].set_xlabel('flavB_max_jet')#'hLeadingFatJetPt')
ax[1].grid()
ax[1].axhline(y=1, linestyle='--')

plt.savefig('flavB_non_tt_st.png', bbox_inches='tight')


print('ratio_values: ', ratio_values)
print('hData.axes[0].edges, ', hData.axes[0].edges)

print('left edge,right edge,data,MC')
for i,x in enumerate(ratio_values):
    print(f'{hData.axes[0].edges[i]},{hData.axes[0].edges[i+1]},{hData_values_toUse[i]},{hBkgTot_values[i]}')

exit()
