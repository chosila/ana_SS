import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob




htlofn = '/afs/cern.ch/work/c/csutanta/HTOAA_CMSSW/analysis/wjethtlo_hist_nofilter/2018/analyze_htoaa_stage1.root'#'/afs/cern.ch/work/c/csutanta/HTOAA_CMSSW/analysis/wjethtlo_hist/2018/analyze_htoaa_stage1.root'
nlofn = '/afs/cern.ch/work/c/csutanta/HTOAA_CMSSW/analysis/wjetnlo_hist_nofilter/2018/analyze_htoaa_stage1.root'

lof = uproot.open(htlofn)['evt/WJetsToLNu_HT_LO/LHE_HT_SR_central']
nlof = uproot.open(nlofn)['evt/WJetsToLNu_Incl_NLO/LHE_HT_SR_central']


loh = lof.to_hist()
loh = loh #[::10j]
loerrors = np.sqrt(loh.variances())
loval, loedges = loh.to_numpy()
locenter = 0.5*(loedges[1:]+loedges[:-1])
binwidth = locenter[1]-locenter[0]

figlo, axlo = plt.subplots()
axlo.bar(locenter, loval, width=binwidth)
axlo.errorbar(locenter, loval, yerr=loerrors)
axlo.set_title('HT LO LHE HT')
axlo.set_xlabel('LHE HT (GeV)')
figlo.savefig('wjet_htlo_lheht.png', bbox_inches='tight')




nloh = nlof.to_hist()
nloh = nloh#[:10j] #[::10j]
nloerrors = np.sqrt(nloh.variances())
nloval, nloedges = nloh.to_numpy()
nlocenter = 0.5*(nloedges[1:]+nloedges[:-1])
binwidth = nlocenter[1]-nlocenter[0]


fignlo, axnlo = plt.subplots(figsize=(30,15))
axnlo.bar(nlocenter, nloval, width=binwidth)
axnlo.set_title('NLO LHE HT')
axnlo.set_xlabel('LHE HT (GeV)')
axnlo.set_ylim([0,800000])
fignlo.savefig('wjet_nlo_lheht.png', bbox_inches='tight')

nloh = nlof.to_hist()
nloh = nloh[::10j]
nloerrors = np.sqrt(nloh.variances())
nloval, nloedges = nloh.to_numpy()
nlocenter = 0.5*(nloedges[1:]+nloedges[:-1])
binwidth = nlocenter[1]-nlocenter[0]


fignlo, axnlo = plt.subplots()
axnlo.bar(nlocenter, nloval, width=binwidth)
axnlo.set_title('NLO LHE HT rebin')
axnlo.set_xlabel('LHE HT (GeV)')
axnlo.set_ylim([0,800000])
fignlo.savefig('wjet_nlo_lheht_rebin.png', bbox_inches='tight')




## log bins

loh = lof.to_hist()
loh = loh # [::10j]
loval, loedges = loh.to_numpy()
rawvals = np.array([])
for freq, edge in zip(loval, loedges[1:]):
    conc = np.ones(round(freq))*edge
    if len(conc) < 1: continue
    rawvals = np.concatenate([rawvals, conc])

print(rawvals)
print(len(rawvals))
logloval = np.log10(rawvals)
logloval = logloval[np.isfinite(logloval)]
fig, ax = plt.subplots()
ax.hist(logloval, bins=100)
ax.set_xlabel('log(LHE HT)')
ax.set_title('LO LOG( LHE HT )')
fig.savefig('wjet_lo_log_lhtht.png', bbox_inches='tight')







# logbinedges = np.unique(np.round(np.logspace(np.log10(70), np.log10(4000), 400)))


# print(loedges.shape)
# print(loval.shape)
# histvals = []
# loedgesshrunk = loedges[:-1]
# ## add all the histogram of bins between bin n and bin n+1 together, add to an array
# for low, high in zip(logbinedges[:-1], logbinedges[1:]):
#     print(low, high)
#     ## identify which element the bin edges that fall in these ranges are
#     #### maybe can be done with mask like
#     #### binHeight = np.sum(loval[(loedges >= low) & (loedges < high)])
#     #### oh heck. loedges and loval are not the same length. maybe we can shrink it somehow
#     #### yes it is shrinkable. We should be able to do loedges[:-1]
#     binheight =  np.sum(loval[(loedgesshrunk >= low) & (loedgesshrunk < high)])
#     histvals.append(binheight)

# fig, ax = plt.subplots()
# centerlabel = 0.5*(logbinedges[1:]+logbinedges[:-1])
# ax.bar(range(len(histvals)), histvals)
# ax.set_xticks(range(len(histvals)))
# ax.set_xticklabels(centerlabel)
# ax.set_title('LO LHE HT (log bins)')
# ax.set_xlabel('LHE HT (GeV)')
# fig.savefig('wjet_nlo_lheht_logbins.png', bbox_inches='tight')
