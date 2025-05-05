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


histograms_dict = {"hLeadingFatJetPt": {sXLabel: 'hLeadingFatJetPt', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 8 }}

## write a way to input the link to the root file from command line and automatically put the resulting plot in the corresponding directory
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('inputFile')

args = parser.parse_args()
print('input file: ', args.inputFile)

sIpFile = args.inputFile
sOpDir_substr = sIpFile.split('/')[-4:-2]
#sOpDir = f'/afs/cern.ch/work/c/csutanta/HTOAA_CMSSW/BBQQ_calibration/plots/btv_presentation/v1/{sOpDir_substr[0]}/{sOpDir_substr[1]}'

sOpDir = '/afs/cern.ch/work/c/csutanta/HTOAA_CMSSW/BBQQ_calibration/plots/v1/calculate_non_top_mc_sf/'

print('output dir: ', sOpDir)

cmsWorkStatus                  = 'Work in Progress'
era                            = '2018'
luminosity_total               = Luminosities_forGGFMode[era][HLT_toUse][0] # 54.54  #59.83

if not os.path.exists(sOpDir):
    os.makedirs(sOpDir)

fIpFile = uproot.open(sIpFile)

def getNonZeroMin(arr):
    min_ = 1e20
    a_   = arr[np.nonzero(arr)]
    if len(a_) > 0:
        min_ = np.min( a_ )
    return min_

def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='lightgrey',
                     edgecolor='none', alpha=0.5):

    # Loop over data points; create box from errors at each point
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
    # matplotlib.patches.Rectangle(xy, width, height, *, angle=0.0, rotation_point='xy', **kwargs)
    #errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
    #              for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]
    errorboxes = [Rectangle((x - xe, y - ye), 2*xe, 2*ye)
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    artists = None
    # Plot errorbars
    #artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
    #                      fmt='none', ecolor=facecolor)

    return artists

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


histo_name = 'hLeadingFatJetPt'
selectionTag = selectionTags[0]
systematic = 'central'
ExpData_list = ['SingleMuon_Run2018A', 'SingleMuon_Run2018B', 'SingleMuon_Run2018C', 'SingleMuon_Run2018D']
sData = 'Data ABCD'

histo_name_toUse = '%s_%s' % (histo_name, selectionTag)

#histo_name_toUse = '%s_%s' % (histo_name, selectionTag)

xAxisRange = histograms_dict[histo_name][sXRange] if sXRange in histograms_dict[histo_name].keys() else None
yAxisRange = histograms_dict[histo_name][sYRange] if sYRange in histograms_dict[histo_name].keys() else None
xAxisLabel = histograms_dict[histo_name][sXLabel] if sXLabel in histograms_dict[histo_name].keys() else None
yAxisLabel = histograms_dict[histo_name][sYLabel] if sYLabel in histograms_dict[histo_name].keys() else None
nRebinX    = histograms_dict[histo_name][sNRebinX] if sNRebinX in histograms_dict[histo_name].keys() else 1
nRebinY    = histograms_dict[histo_name][sNRebinY] if sNRebinY in histograms_dict[histo_name].keys() else 1

nHistoDimemsions = None
yAxisRange_cal      = [1e20, 10]
yRatioAxisRange_cal = [1e20, 10]
xError = np.array([])
hData = None
hBkgTot_values = None
hBkgTot_variance = None
hStack_values_list = np.array([])
hStack_edges = np.array([])
hStack_centers = np.array([])
sStack_list = []
nBkgTot = 0
significanceAvg = []


hBkg_list = []
sBkg_list = []
hBkg_integral_list = []

#groups = [180] + [16]*20 + [32]*6 + [48]*2 + [96]*7 + [40]
groups = [180] + [16]*20 + [32]*6 + [48]*2 + [96]*6 + [136]
# groups = [180] + [32]*16 + [48]*2 + [96]*6 + [136]
#groups = [180] + [48]*13 + [96]*6 + [120]

for dataset in MCBkg_list:
    histo_name_toUse_full = 'evt/%s/%s_%s' % (dataset, histo_name_toUse, systematic)
    print(f"{histo_name_toUse_full = }")
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

    if nHistoDimemsions == 1:
        yMin_ = getNonZeroMin(h.values())
        yMax_ = np.max(h.values())
        if (yMin_ < yAxisRange_cal[0]):
            yAxisRange_cal[0] = yMin_
        if (yMax_ > yAxisRange_cal[1]):
            yAxisRange_cal[1] = yMax_

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
hStack_variance_list  = np.array( [ h.variances() for h in hStack_list ] )
hStack_error_list     = np.array( [ np.sqrt(h.variances()) for h in hStack_list ] )

hStack_edges          = hStack_list[0].axes[0].edges
hStack_centers        = hStack_list[0].axes[0].centers
xError                = (hStack_list[0].axes[0].edges[1:] - hStack_list[0].axes[0].edges[0:-1]) / 2 if len(xError) == 0 else xError

hBkgTot_values        = np.sum(hStack_values_list, axis=0)
hBkgTot_variance      = np.sum(hStack_variance_list, axis=0)


nHists = len(MCBkg_list)
colors_toUse = [ colors_bkg_list[i][0] for i in range(nHists) ]
alpha_toUse  = [ colors_bkg_list[i][1] for i in range(nHists) ]
hatch_toUse  = [ colors_bkg_list[i][2] for i in range(nHists) ]

hData = None
for ExpData_component in ExpData_list:
    histo_name_toUse_full = 'evt/%s/%s_%s' % (ExpData_component, histo_name_toUse, systematics_forData)
    h = fIpFile[histo_name_toUse_full].to_hist()
    if hData == None: hData = h
    else:             hData = hData + h


rebin  = hist.rebin(groups=groups)
hData  = hData[::rebin]

xError = (hData.axes[0].edges[1:] - hData.axes[0].edges[0:-1]) / 2


##
# print(hStack_list)
# print(sStack_list)
# print(hData)


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
    if False: #('TTTo' in hname) or ('SingleTop' in hname):

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

colors_toUse = [ colors_bkg_list[i][0] for i in range(len(hStack_list_other)) ]
alpha_toUse  = [ colors_bkg_list[i][1] for i in range(len(hStack_list_other)) ]
hatch_toUse  = [ colors_bkg_list[i][2] for i in range(len(hStack_list_other)) ]

fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(8,10), sharex='col', gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
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
#ax[0].set_ylim(yAxisRange_cal[0], yAxisRange_cal[1]*1.5)
ax[0].set_ylim(min(hData_values_toUse), max(hData_values_toUse)*1.8)
ax[0].set_xlim(180, 1500)
ax[0].legend(fontsize=12, ncol=2)
ax[1].set_ylim(-.1, 2.3)
ax[1].set_xlabel('hLeadingFatJetPt')
ax[1].grid()
ax[1].axhline(y=1, linestyle='--')

plt.savefig('non_tt_st_data_vs_mc.png', bbox_inches='tight')


print('ratio_values: ', ratio_values)
print('hData.axes[0].edges, ', hData.axes[0].edges)

print('left edge,right edge,center,ratio values')
for i,x in enumerate(ratio_values):
    print(f'{hData.axes[0].edges[i]},{hData.axes[0].edges[i+1]},{(hData.axes[0].edges[i]+hData.axes[0].edges[i+1])/2},{x}')

exit()

## ========================================================================================================================================== original
from HistogramListForPlottingDataVsMC_TriggerStudy_GGFMode import *
for sData, ExpData_list in ExpData_dict.items():
    luminosity_toUse = 0
    for ExpData_component in ExpData_list:
        DatasetEra_         = ExpData_component.split(era)[1][0] # 'JetHT_Run2018A'.split('2018')[1][0]
        luminosity_forEra_  = Luminosities_forGGFMode_perEra[era][HLT_toUse][DatasetEra_]
        luminosity_toUse   += luminosity_forEra_
        #print(f"{ExpData_list = }, {DatasetEra_ = }, {luminosity_forEra_ = } ")
    luminosity_Scaling_toUse = round(luminosity_toUse, 2) / round(luminosity_total, 2)
    luminosity_toUse = round(luminosity_toUse, 2)
    #print(f"{sData}: {ExpData_list}, {luminosity_toUse = }, {luminosity_total = },  {luminosity_Scaling_toUse = }")

    for selectionTag in selectionTags:
        #dataBlindOption_toUse = dataBlindOption if selectionTag != 'SR' else DataBlindingOptions.BlindPartially

        for histo_name in ['hLeadingFatJetPt']:#histograms_dict.keys():
            histo_name_toUse = '%s_%s' % (histo_name, selectionTag)
            for systematic in systematics_list:
                #systematic = 'central'
                #selectionTag = 'sel_0b_BBQQ'
                #print(histo_name)

                for yAxisScale in ['linearY']: # ['linearY', 'logY']
                    xAxisRange = histograms_dict[histo_name][sXRange] if sXRange in histograms_dict[histo_name].keys() else None
                    yAxisRange = histograms_dict[histo_name][sYRange] if sYRange in histograms_dict[histo_name].keys() else None
                    xAxisLabel = histograms_dict[histo_name][sXLabel] if sXLabel in histograms_dict[histo_name].keys() else None
                    yAxisLabel = histograms_dict[histo_name][sYLabel] if sYLabel in histograms_dict[histo_name].keys() else None

                    nHistoDimemsions = None
                    yAxisRange_cal      = [1e20, 10]
                    yRatioAxisRange_cal = [1e20, 10]
                    xError = np.array([])
                    hData = None
                    hBkgTot_values = None
                    hBkgTot_variance = None
                    hStack_values_list = np.array([])
                    hStack_edges = np.array([])
                    hStack_centers = np.array([])
                    sStack_list = []
                    nBkgTot = 0
                    significanceAvg = [] #np.array([])
                    h_ttbar_st = []


                    print(f"\n\n {histo_name_toUse = }, {systematic = }, {yAxisScale = }, ")

                    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(8,10), sharex='col', gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})


                ## need this to calculate the total of the data
                hData = None
                for ExpData_component in ExpData_list:
                    histo_name_toUse_full = 'evt/%s/%s_%s' % (ExpData_component, histo_name_toUse, systematics_forData)
                    h = fIpFile[histo_name_toUse_full].to_hist()
                    if hData == None:
                        hData = h
                    else:
                        hData = hData + h

                data_sum_total = hData.sum().value


                if len(MCBkg_list) > 0:
                    hBkg_list = []
                    sBkg_list = []
                    hBkg_integral_list = []
                    for dataset in MCBkg_list:
                        histo_name_toUse_full = 'evt/%s/%s_%s' % (dataset, histo_name_toUse, systematic)
                        print(f"{histo_name_toUse_full = }")
                        h = fIpFile[histo_name_toUse_full].to_hist()
                        nHistoDimemsions = len(h.axes)
                        if nHistoDimemsions == 2 and yAxisScale == 'logY': break  # No need to plot 2-D hist with logY
                        #rint(luminosity_Scaling_toUse)
                        # = h * luminosity_Scaling_toUse
                        ## scale histogram by 0.6 if bb/bbq/bbqq
                        if ('TTToSemiLeptonic_powheg_bb' in histo_name_toUse_full) or ('TTTo2L2Nu_powheg_bb' in histo_name_toUse_full):
                            h=h*0.6

                        print(h)
                        groups = [180] + [16]*20 + [32]*6 + [48]*2 + [96]*7 + [40]
                        rebin = hist.rebin(groups=groups)
                        h = h[::rebin]

                        ## check if ttbar or singletop, add to a list, continue loop so it doesn't get added to the hbkg list

                        # if ('TTTo' in histo_name_toUse_full) or ('SingleTop' in histo_name_toUse_full):
                            #h_ttbar_st.append(h)
                            #continue

                        nTot_ = h.values().sum()
                        hBkg_list.append(h)
                        sBkg_list.append(dataset)
                        hBkg_integral_list.append(nTot_)


                        if abs(nTot_ - 0) < 1e-10: continue

                        if nHistoDimemsions == 1:
                            yMin_ = getNonZeroMin(h.values())
                            yMax_ = np.max(h.values())
                            if (yMin_ < yAxisRange_cal[0]):
                                yAxisRange_cal[0] = yMin_
                            if (yMax_ > yAxisRange_cal[1]):
                                yAxisRange_cal[1] = yMax_



                    # sort histograms in decreasing yield
                    isReverseSortForStack = True
                    idx_hBkg_sortedByIntegral = sorted(range(len(hBkg_integral_list)), key=lambda i: hBkg_integral_list[i], reverse=isReverseSortForStack)

                    sf_to_data_total = data_sum_total / np.sum(hBkg_integral_list)
                    hStack_list = [ hBkg_list[idx]*sf_to_data_total for idx in idx_hBkg_sortedByIntegral ]
                    sStack_list = [ sBkg_list[idx] for idx in idx_hBkg_sortedByIntegral ]

                    hStack_values_list    = np.array( [ h.values() for h in hStack_list ] )
                    hStack_variance_list  = np.array( [ h.variances() for h in hStack_list ] )

                    hBkgTot_values        = np.sum(hStack_values_list, axis=0)
                    hBkgTot_variance      = np.sum(hStack_variance_list, axis=0)

                    hStack_edges          = hStack_list[0].axes[0].edges
                    hStack_centers        = hStack_list[0].axes[0].centers
                    xError                = (hStack_list[0].axes[0].edges[1:] - hStack_list[0].axes[0].edges[0:-1]) / 2 if len(xError) == 0 else xError


                    # Update yRange for hStackBkg -------
                    if nHistoDimemsions == 1:
                        yMin_ = getNonZeroMin(hBkgTot_values)
                        yMax_ = np.max(hBkgTot_values)
                    if yMin_ < yAxisRange_cal[0]:
                        yAxisRange_cal[0] = yMin_
                    if (yMax_ > yAxisRange_cal[1]):
                        yAxisRange_cal[1] = yMax_

                    nHists = len(hStack_list)#len(MCBkg_list)
                    colors_toUse = [ colors_bkg_list[i][0] for i in range(nHists) ]
                    alpha_toUse  = [ colors_bkg_list[i][1] for i in range(nHists) ]
                    hatch_toUse  = [ colors_bkg_list[i][2] for i in range(nHists) ]


                    ## shortening some labels in sStack_list
                    renamed_sStack_list = []
                    for n in sStack_list:
                        newn = n.replace('Hadronic', 'Had')
                        newn = newn.replace('SemiLeptonic', '1L1Nu')
                        newn = newn.replace('_Incl', '')
                        newn = newn.replace('_powheg', '')
                        renamed_sStack_list.append(newn)

                    # https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_style_reference.html

                    if nHistoDimemsions == 1: # 1-D histogram
                        hep.histplot(
                            hStack_values_list,
                            bins=hStack_edges,
                            ax=ax[0],
                            histtype='fill',
                            stack=True,
                            label=renamed_sStack_list,
                            color=colors_toUse,
                            alpha=alpha_toUse,
                            hatch=hatch_toUse,
                            sort='yield'
                        )


                    # plot total background
                    #hep.histplot(hBkgTot_values, bins=hStack_edges, ax=ax, yerr=np.sqrt(hBkgTot_variance), histtype='errorbar', color='grey', label='Total background')

                    # plot totoal background error bars only
                    make_error_boxes(
                        ax=ax[0],
                        xdata=hStack_centers,
                        ydata=hBkgTot_values,
                        xerror=xError,
                        yerror=np.sqrt(hBkgTot_variance),
                        facecolor='grey',
                        edgecolor='none',
                        alpha=0.5
                    )

                elif nHistoDimemsions == 2 and 1==0: # 2-D histogram
                    #print(f"{list(hStack_list[0].values()) = }, \n{hStack_list[0].variances() = }, ")
                    #print(f"{getNonZeroMin(h.values()) = }")
                    hep.hist2dplot(
                        hBkgTot_values,
                        xbins=hStack_list[0].axes[0].edges,
                        ybins=hStack_list[0].axes[1].edges,
                        #labels='Bkg_total',
                        cmin=getNonZeroMin(hStack_list[0].values()),
                        ax=ax[0]
                    )

                    # No. of events in total background
                    nBkgTot = np.sum(hBkgTot_values)



                    if True:
                        hData = None
                        for ExpData_component in ExpData_list:
                            histo_name_toUse_full = 'evt/%s/%s_%s' % (ExpData_component, histo_name_toUse, systematics_forData)
                            h = fIpFile[histo_name_toUse_full].to_hist()
                            if hData == None:
                                hData = h
                            else:
                                hData = hData + h


                        groups = [180] + [16]*20 + [32]*6 + [48]*2 + [96]*7 + [40]
                        rebin = hist.rebin(groups=groups)
                        hData = hData[::rebin]

                        xError = (hData.axes[0].edges[1:] - hData.axes[0].edges[0:-1]) / 2

                        if nHistoDimemsions == 1:
                            yMin_ = getNonZeroMin(hData.values())
                            yMax_ = np.max(hData.values())
                            if yMin_ < yAxisRange_cal[0]:
                                yAxisRange_cal[0] = yMin_
                            if yMax_ > yAxisRange_cal[1]:
                                yAxisRange_cal[1] = yMax_
                            #print(f"Data: {yMin_ = }, {yMin_}")


                        ## subtract the ttbar from hdata?
                        hData_values_toUse = hData.values()
                        print(type(hData))
                        print(type(h_ttbar_st[0]))
                        #for h_MC in h_ttbar_st:
                        #    hData_values_toUse = hData_values_toUse - h_MC.values()


                        print(hData_values_toUse)
                        print(hData.values())
                        hData_errors_toUse = np.sqrt(hData.variances())


                        if nHistoDimemsions == 1:
                            #hep.histplot(hData.values(), bins=hData.axes[0].edges, ax=ax[0], yerr=np.sqrt(hData.variances()), histtype='errorbar', color='black', label='Data')
                            hep.histplot(
                                hData_values_toUse,
                                bins=hData.axes[0].edges,
                                ax=ax[0],
                                yerr=hData_errors_toUse,
                                histtype='errorbar',
                                color='black',
                                label='%s %s' % (sData, ' ')
                                )


                        # Ratio plot ---------------------------------------------------------
                        ratio_values = np.divide(hData_values_toUse, hBkgTot_values, where=hBkgTot_values!=0, out=np.ones(hData.shape))
                        ratio_error  = hData_errors_toUse
                        ratio_error  = np.divide(ratio_error, hBkgTot_values, where=hBkgTot_values!=0, out=np.zeros(hData.shape))
                        ratio_syst   = np.sqrt(hBkgTot_variance)
                        ratio_syst   = np.divide(ratio_syst, hBkgTot_values, where=hBkgTot_values!=0, out=np.zeros(hData.shape))

                        # print(f"ratio_values ({ratio_values.shape}): {ratio_values}")
                        if nHistoDimemsions == 1:
                            yMin_ = getNonZeroMin( ratio_values - ratio_error)
                            yMax_ = np.max( ratio_values + ratio_error)
                            if yMin_ < yRatioAxisRange_cal[0]:
                                yRatioAxisRange_cal[0] = yMin_
                            if yMax_ > yRatioAxisRange_cal[1]:
                                yRatioAxisRange_cal[1] = yMax_

                        if nHistoDimemsions == 1:
                            hep.histplot(
                                ratio_values,
                                bins=hData.axes[0].edges,
                                ax=ax[1],
                                yerr=ratio_error,
                                histtype='errorbar',
                                color='black',
                                label='Data'
                                )
                            #if xAxisRange: ax[1].set_xlim(xAxisRange[0], xAxisRange[1])

                            # plot totoal background error bars only for ratio plot
                            make_error_boxes(
                                ax=ax[1],
                                xdata=hData.axes[0].centers,
                                ydata=np.full(len(hData.axes[0].centers), 1),
                                xerror=xError,
                                yerror=ratio_syst,
                                facecolor='grey',
                                edgecolor='none',
                                alpha=0.5
                                )


                    # Upper plot cosmetics ---------
                    if xAxisRange: ax[0].set_xlim(xAxisRange[0], xAxisRange[1])
                    print(f"\nAt the end {yAxisRange_cal = }")

                    if yAxisRange: ax[0].set_ylim(yAxisRange[0], yAxisRange[1])
                    elif nHistoDimemsions == 1:

                        print(f'{yAxisRange_cal=}')

                        yMaxOffset = 10**(math.log10(yAxisRange_cal[1] / abs(yAxisRange_cal[0])) * 0.4) if ((yAxisScale == 'logY')) else 1.6


                        if yAxisScale == 'logY':
                            yAxisRange_cal[0] = abs(yAxisRange_cal[0]) * logYMinScaleFactor
                            yAxisRange_cal[1] = pow(yAxisRange_cal[1], 1.8) ## make top 1.5 times larger to make space for margins

                        else:
                            yAxisRange_cal[0] = yAxisRange_cal[0]
                            yAxisRange_cal[1] = yAxisRange_cal[1]*1.8

                        print(f"\nAt the end updated {yAxisRange_cal = } \t {yAxisScale = }")
                        ax[0].set_ylim(yAxisRange_cal[0], yAxisRange_cal[1]) ## ylim about 1.5x the max value so there's space for legend


                    if xAxisLabel:
                        ax[0].set_xlabel(xAxisLabel)

                    if yAxisLabel:
                        ax[0].set_ylabel(yAxisLabel)

                    ax[0].legend(fontsize=12, loc='upper right', ncol=2)


                    if yAxisScale == 'logY':
                        ax[0].set_yscale('log', base=10)

                    # Ratio plot cosmetics ---------
                    if yRatioAxisRange_cal[0] < yRatioLimit[0]: yRatioAxisRange_cal[0] = yRatioLimit[0]
                    if yRatioAxisRange_cal[1] > yRatioLimit[1]: yRatioAxisRange_cal[1] = yRatioLimit[1]
                    yRatioAxisRange_cal_maxDeviation = max(abs(yRatioAxisRange_cal[0] - 1), abs(yRatioAxisRange_cal[1] - 1))
                    yRatioAxisRange_cal[0] = 1 - yRatioAxisRange_cal_maxDeviation
                    yRatioAxisRange_cal[1] = 1 + yRatioAxisRange_cal_maxDeviation
                    if xAxisRange: ax[1].set_xlim(xAxisRange[0], xAxisRange[1])
                    ax[1].set_ylim(yRatioAxisRange_cal[0], yRatioAxisRange_cal[1])
                    #ax1[1].set_ylim(yRatioAxisRange_cal[0], yRatioAxisRange_cal[1])
                    print(f"{yRatioAxisRange_cal = }")

                    if xAxisLabel:
                        ax[1].set_xlabel(xAxisLabel)

                    ax[1].set_ylabel('Data/MC')


                    ax[1].axhline(y=1, linestyle='--')

                    ax[1].grid()




                    isData = True#True if dataBlindOption_toUse != DataBlindingOptions.BlindFully else False
                    fontsize_toUse = 18 if isData else 15
                    hep.cms.label(ax=ax[0], data=isData, year=era, lumi=luminosity_toUse, label=cmsWorkStatus, fontsize=fontsize_toUse)


                    ax[0].text(0.6, 0.63, selectionTag,
                            fontsize=12, fontstyle='italic',
                                horizontalalignment='center',
                                verticalalignment='center',
                                transform=ax[0].transAxes
                                )


                    fig.savefig(f'original.png', dpi=80, bbox_inches='tight')
                    #fig.savefig('%s/%s_%s_%s_%s.png' % (sOpDir,histo_name_toUse,systematic,sData, yAxisScale), transparent=False, dpi=80, bbox_inches="tight")

                    plt.close('all')

# %%
