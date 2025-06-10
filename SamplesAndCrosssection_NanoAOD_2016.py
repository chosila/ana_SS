from collections import OrderedDict as OD

sXS           = "xs"
sNameSp       = "nameSp"
sNanoSkimv2   = "skim_v2"
sNEvents      = "nEvents"
sSumEvents    = "sumEvents"
sNEvtSkimv2   = "%s_%s" % (sNanoSkimv2, sNEvents)
sSumEvtSkimv2 = "%s_%s" % (sNanoSkimv2, sSumEvents)

list_datasetAndXs_2016 = OD([

    ## TTbar - NLO powheg
    ("/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",                   {sXS: 380.133 , sNEvtSkimv2: 109380000, sSumEvtSkimv2: 108494782}),
    ("/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",               {sXS: 364.328 , sNEvtSkimv2: 144974000, sSumEvtSkimv2: 143804034}),
    ("/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",                      {sXS:  87.339 , sNEvtSkimv2:  43630000, sSumEvtSkimv2:  43277246}),


    ## ST NLO
    ("/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",     {sXS: 134.2   , sNEvtSkimv2: 55783000, sSumEvtSkimv2: 55461204}),
    ("/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",   {sXS:  80.0   , sNEvtSkimv2: 29394000, sSumEvtSkimv2: 29233704}),
    ("/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",            {sXS:  39.65  , sNEvtSkimv2:  2491000, sSumEvtSkimv2:  2490860}),
    ("/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",        {sXS:  39.65  , sNEvtSkimv2:  2554000, sSumEvtSkimv2:  2553882}),
    ("/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",      {sXS:  21.63  , sNEvtSkimv2:  3368375, sSumEvtSkimv2:  3368237}),
    ("/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",  {sXS:  21.63  , sNEvtSkimv2:  3654510, sSumEvtSkimv2:  3654330}),
    ("/ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",        {sXS:   5.041 , sNEvtSkimv2:  5300000, sSumEvtSkimv2:  3448806}),
    ("/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",          {sXS:   4.831 , sNEvtSkimv2:  5471000, sSumEvtSkimv2:  3562866}),


    # DYJetsToLL LO Incl
    ("/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",          {sXS: 18610    , sNEvtSkimv2: 23706672, sSumEvtSkimv2: 23706672}),
    ("/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",              {sXS:  6077.22 , sNEvtSkimv2: 82448537, sSumEvtSkimv2: 82448537}),


    ## WJetsToLNu
    ("/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",         {sXS:  1440.0     , sNEvtSkimv2: 20618288, sSumEvtSkimv2: 20618288}),
    ("/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",        {sXS:  1431.0     , sNEvtSkimv2: 20479332, sSumEvtSkimv2: 20479332}),
    ("/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",        {sXS:   382.1     , sNEvtSkimv2: 18440960, sSumEvtSkimv2: 18440960}),
    ("/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",        {sXS:    51.54    , sNEvtSkimv2:  2934047, sSumEvtSkimv2:  2934047}),
    ("/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",        {sXS:    12.49    , sNEvtSkimv2:  4519055, sSumEvtSkimv2:  4519055}),
    ("/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",       {sXS:     5.619   , sNEvtSkimv2:  4448546, sSumEvtSkimv2:  4448546}),
    ("/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",      {sXS:     1.321   , sNEvtSkimv2:  3902685, sSumEvtSkimv2:  3902685}),
    ("/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",       {sXS:     0.02992 , sNEvtSkimv2:  4287657, sSumEvtSkimv2:  4287657}),

    ## SingleMuon
    # XS (cross-section) does not matter for data sample
    ("/SingleMuon/Run2016B-ver2_HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD", {sXS: -1}),
    ("/SingleMuon/Run2016C-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD", {sXS: -1}),
    ("/SingleMuon/Run2016D-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD", {sXS: -1}),
    ("/SingleMuon/Run2016E-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD", {sXS: -1}),
    ("/SingleMuon/Run2016F-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD", {sXS: -1}),
    ("/SingleMuon/Run2016F-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD", {sXS: -1}),
    ("/SingleMuon/Run2016G-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD", {sXS: -1}),
    ("/SingleMuon/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD", {sXS: -1}),
    #("/SingleMuon/Run2017G-UL2017_MiniAODv2_NanoAODv9_GT36-v2/NANOAOD", {sXs: -1}), ## these are low PU that are not used in most analysis : https://twiki.cern.ch/twiki/bin/viewauth/CMS/LumiRecommendationsRun2#2017
    #("/SingleMuon/Run2017H-UL2017_MiniAODv2_NanoAODv9_GT36-v1/NANOAOD", {sXs: -1}),

    ## EGamma
    # XS (cross-section) does not matter for data sample
    ("/SingleElectron/Run2016B-ver2_HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD", {sXS: -1}),
    ("/SingleElectron/Run2016C-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD", {sXS: -1}),
    ("/SingleElectron/Run2016D-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD", {sXS: -1}),
    ("/SingleElectron/Run2016E-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD", {sXS: -1}),
    ("/SingleElectron/Run2016F-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD", {sXS: -1}),
    ("/SingleElectron/Run2016F-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD", {sXS: -1}),
    ("/SingleElectron/Run2016G-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD", {sXS: -1}),
    ("/SingleElectron/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD", {sXS: -1}),
])
