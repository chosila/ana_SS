# %%
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

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
from sklearn import metrics

from HistogramListForPlotting_ScaleFactor import *
# from HistogramListForPlottingDataVsMC_TriggerStudy_GGFMode import *
#from HistogramListForPlottingDataVsMC_Analysis_GGFMode import *
#from HistogramListForPlottingDataVsMC_Analysis_Example import *

sys.path.insert(1, '../') # to import file from other directory (../ in this case)

from htoaa_Settings import *

class DataBlindingOptions(enum.Enum):
    BlindPartially = '(partially blind)'
    BlindFully     = '(blind)'
    Unblind        = ' '


#sIpFile = '/eos/cms/store/user/ssawant/htoaa/analysis/20230831_SelPNetMDXbbNSV/2018/analyze_htoaa_stage1.root'
#sOpDir  = '/eos/cms/store/user/ssawant/htoaa/analysis/20230831_SelPNetMDXbbNSV/2018/plots'
#sIpFile = '/eos/cms/store/user/ssawant/htoaa/analysis/20230921_SFPNetMDXbbvsQCD/2018/analyze_htoaa_stage1.root'
#sOpDir  = '/eos/cms/store/user/ssawant/htoaa/analysis/20230921_SFPNetMDXbbvsQCD/2018/plots'
#sIpFile = '/eos/cms/store/user/ssawant/htoaa/analysis/20230922_DataSplitByEra/2018/analyze_htoaa_stage1.root'
#sOpDir  = '/eos/cms/store/user/ssawant/htoaa/analysis/20230922_DataSplitByEra/2018/plots'
#sIpFile = '/eos/cms/store/user/ssawant/htoaa/analysis/20231019_PNetTaggerSignScan/2018/analyze_htoaa_stage1.root'
#sOpDir  = '/eos/cms/store/user/ssawant/htoaa/analysis/20231019_PNetTaggerSignScan/2018/plots2'
#sIpFile =  '/afs/cern.ch/work/c/csutanta/HTOAA_CMSSW/analysis/singleLep_fixed/2018/analyze_htoaa_stage1.root' #
#sIpFile = '/afs/cern.ch/work/c/csutanta/HTOAA_CMSSW/analysis/unskimmed_singlelep/2018/analyze_htoaa_stage1.root'
#sIpFile = '/afs/cern.ch/work/c/csutanta/HTOAA_CMSSW/analysis/unskimmed_singlelep_bdtScoreCut/2018/analyze_htoaa_stage1.root'
#sOpDir  = '/afs/cern.ch/work/c/csutanta/HTOAA_CMSSW/htoaa/plots/BBROC/bdtScoreCut' # '/afs/cern.ch/work/c/csutanta/HTOAA_CMSSW/htoaa/plots/unskimmed_singleLep' #'/eos/cms/store/user/ssawant/htoaa/analysis/20231019_PNetMD_Hto4b_Htoaa4bOverQCD_WP60/2018/plots2'
#sIpFile = '/afs/cern.ch/work/c/csutanta/HTOAA_CMSSW/analysis/unskimmed_singlelep_nobdtScoreCut/2018/analyze_htoaa_stage1.root'
sIpFile = '/afs/cern.ch/work/c/csutanta/HTOAA_CMSSW/analysis/unskimmed_singlelep_bdtScoreCutless3/2018/analyze_htoaa_stage1.root'
sOpDir  = '/afs/cern.ch/work/c/csutanta/HTOAA_CMSSW/htoaa/plots/BBROC/bdtScoreCutless3'


cmsWorkStatus                  = 'Work in Progress'
era                            = '2018'
luminosity_total               = Luminosities_forGGFMode[era][HLT_toUse][0] # 54.54  #59.83
dataBlindOption                = DataBlindingOptions.Unblind # DataBlindingOptions.BlindPartially , DataBlindingOptions.BlindFully , DataBlindingOptions.Unblind
significantThshForDataBlinding = 0.125 # blind data in bins with S/sqrt(B) > significantThshForDataBlinding while running with dataBlindOption = DataBlindingOptions.BlindPartially


if not os.path.exists(sOpDir):
    os.makedirs(sOpDir)

fIpFile = uproot.open(sIpFile)




for sData, ExpData_list in ExpData_dict.items():
    luminosity_toUse = 0
    for ExpData_component in ExpData_list:
        DatasetEra_         = ExpData_component.split(era)[1][0] # 'JetHT_Run2018A'.split('2018')[1][0]
        luminosity_forEra_  = Luminosities_forGGFMode_perEra[era][HLT_toUse][DatasetEra_]
        luminosity_toUse   += luminosity_forEra_
        print(f"{ExpData_list = }, {DatasetEra_ = }, {luminosity_forEra_ = } ")
    luminosity_Scaling_toUse = round(luminosity_toUse, 2) / round(luminosity_total, 2)
    luminosity_toUse = round(luminosity_toUse, 2)
    print(f"{sData}: {ExpData_list}, {luminosity_toUse = }, {luminosity_total = },  {luminosity_Scaling_toUse = }")

    for selectionTag in selectionTags:

        for histo_name in histograms_dict.keys():
            predictedValues = []
            expectedValues_bbFromTop = []
            expectedValues_nBHadron = []
            histo_name_toUse = '%s_%s' % (histo_name, selectionTag)
            for systematic in systematics_list:
                print(f"\n\n {histo_name_toUse = }, {systematic = }")


                hBkg_list = []
                sBkg_list = []
                hBkg_integral_list = []

                bb_h = []
                nonbb_h = []
                b_h = []
                ob_h = []
                bbq_bbqq = []

                for dataset in MCBkg_list:
                    histo_name_toUse_full = 'evt/%s/%s_%s' % (dataset, histo_name_toUse, systematic)
                    nBHadrons_histo_name = 'evt/%s/hLeadingFatJet_nBHadrons_%s_%s' % (dataset, selectionTag, systematic)

                    print(f'{histo_name_toUse_full=}')

                    h = fIpFile[histo_name_toUse_full].to_hist()
                    ## rebin to avoid empty bins
                    h = h[::2j]
                    # nBHadrons_h = fIpFile[nBHadrons_histo_name].to_hist()
                    h = h * luminosity_Scaling_toUse

                    if "TTT" and "bb" in dataset:
                        bb_h.append(h.values())
                    elif ("TTT" and "bq" in dataset) or ('TTT' and '1b' in dataset):
                        b_h.append(h.values())
                        nonbb_h.append(h.values())

                    elif ('TTT' and '0b') in dataset:
                        ob_h.append(h.values())
                        nonbb_h.append(h.values())

                    else:
                        nonbb_h.append(h.values())

                    if ("TTT" and "bbq") in dataset:
                        bbq_bbqq.append(h.values())



                sumaxis=0
                bb_h = np.array(bb_h).sum(axis=sumaxis)
                b_h = np.array(b_h).sum(axis=sumaxis)
                ob_h = np.array(ob_h).sum(axis=sumaxis)
                nonbb_h = np.array(nonbb_h).sum(axis=sumaxis)
                bbq_bbqq = np.array(bbq_bbqq).sum(axis=sumaxis)


                for stackedvals, stackedname in [[bb_h, 'bb'], [b_h, 'one_b'], [ob_h, 'zero_b'], [nonbb_h, 'nonbb'], [bbq_bbqq, 'bbq_bbqq']]:
                    fig, ax = plt.subplots()
                    ax.bar(np.linspace(0,1,100),stackedvals, width=(1/100))
                    ax.set_title(stackedname+histo_name)
                    fig.savefig(f'{sOpDir}/stacked_{histo_name}_{stackedname}.png')
                    plt.close(fig)

                for bbh, nonbbh, savename in [[bb_h, nonbb_h, 'bb_vs_nonbb'],
                                              [bb_h, b_h, 'bb_vs_1b'],
                                              [bb_h, ob_h, 'bb_vs_0b'],
                                              [bbq_bbqq, nonbb_h, 'bbq+bbqq_vs_nonbb']]:
                    fpr = (nonbbh.sum() - np.cumsum(nonbbh)) / nonbbh.sum()
                    tpr = (bbh.sum() - np.cumsum(bbh)) / bbh.sum()
                    print(savename, histo_name, selectionTag)

                    roc_auc = metrics.auc(fpr,tpr)
                    print(fpr)
                    #if 'XbbOverQCD' in histo_name: break
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr , label = 'AUC = %.2f' % roc_auc)
                    ax.legend()
                    ax.grid()
                    ax.set_xlim([0,1])
                    ax.set_ylim([0,1])
                    ax.plot([0,1], [0,1], 'r--')
                    ax.set_xlabel('fpr')
                    ax.set_ylabel('tpr')
                    ax.set_title(f'{savename} {histo_name} ROC')
                    fig.savefig(f'{sOpDir}/{savename}_ROC_{histo_name}_{selectionTag}.png')
                    plt.close(fig)
