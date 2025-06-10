from collections import OrderedDict as OD

sXS           = "xs"
sNameSp       = "nameSp"
sNanoSkimv2   = "skim_v2"
sNEvents      = "nEvents"
sSumEvents    = "sumEvents"
sNEvtSkimv2   = "%s_%s" % (sNanoSkimv2, sNEvents)
sSumEvtSkimv2 = "%s_%s" % (sNanoSkimv2, sSumEvents)

list_datasetAndXs_2017 = OD([

    ## TTbar - NLO powheg

    ("/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",                              {sXS: 380.133 , sNEvtSkimv2: 235719999, sSumEvtSkimv2: 231910835}),
    ("/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",               {sXS: 364.328 , sNEvtSkimv2: 355332000, sSumEvtSkimv2: 352462632}),
    ("/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",                      {sXS:  87.339 , sNEvtSkimv2: 106724000, sSumEvtSkimv2: 105859990}),


    ## ST NLO
    ("/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",     {sXS: 134.2   , sNEvtSkimv2: 142078000, sSumEvtSkimv2: 141260698}),
    ("/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM", {sXS:  80.0   , sNEvtSkimv2: 70203000 , sSumEvtSkimv2: 69821018 }),
    ("/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",            {sXS:  39.65  , sNEvtSkimv2: 5649000,   sSumEvtSkimv2: 5648712  }),
    ("/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",        {sXS:  39.65  , sNEvtSkimv2: 5674000,   sSumEvtSkimv2: 5673700  }),
    ("/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",      {sXS:  21.63  , sNEvtSkimv2: 8507203,   sSumEvtSkimv2: 8506765  }),
    ("/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",  {sXS:  21.63  , sNEvtSkimv2: 8433998,   sSumEvtSkimv2: 8433562  }),
    ("/ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",        {sXS:   5.041 , sNEvtSkimv2: 11696999,  sSumEvtSkimv2: 7615207  }),
    ("/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",          {sXS:   4.831 , sNEvtSkimv2: 13882000,  sSumEvtSkimv2: 9037288  }),


    # DYJetsToLL LO Incl
    ("/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",                     {sXS: 18610    , sNEvtSkimv2: 68480179,  sSumEvtSkimv2: 68480179}),
    ("/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",              {sXS:  6077.22 , sNEvtSkimv2: 103344974, sSumEvtSkimv2: 103344974}),



    ## WJetsToLNu
    ("/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",         {sXS:  1440.0     , sNEvtSkimv2: 44736228 , sSumEvtSkimv2: 44736228 }),
    ("/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",        {sXS:  1431.0     , sNEvtSkimv2: 47424468 , sSumEvtSkimv2: 47424468 }),
    ("/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",        {sXS:   382.1     , sNEvtSkimv2: 42602407 , sSumEvtSkimv2: 42602407 }),
    ("/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",        {sXS:    51.54    , sNEvtSkimv2: 5468473 ,  sSumEvtSkimv2: 5468473 }),
    ("/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",        {sXS:    12.49    , sNEvtSkimv2: 5545298 ,  sSumEvtSkimv2: 5545298 }),
    ("/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v3/NANOAODSIM",       {sXS:     5.619   , sNEvtSkimv2: 5088483 ,  sSumEvtSkimv2: 5088483 }),
    ("/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",      {sXS:     1.321   , sNEvtSkimv2: 4955636 ,  sSumEvtSkimv2: 4955636 }),
    ("/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",       {sXS:     0.02992 , sNEvtSkimv2: 1185699 ,  sSumEvtSkimv2: 1185699 }),

    ## SingleMuon
    # XS (cross-section) does not matter for data sample
    ("/SingleMuon/Run2017B-UL2017_MiniAODv2_NanoAODv9_GT36-v1/NANOAOD", {sXS: -1}),
    ("/SingleMuon/Run2017C-UL2017_MiniAODv2_NanoAODv9_GT36-v1/NANOAOD", {sXS: -1}),
    ("/SingleMuon/Run2017D-UL2017_MiniAODv2_NanoAODv9_GT36-v1/NANOAOD", {sXS: -1}),
    ("/SingleMuon/Run2017E-UL2017_MiniAODv2_NanoAODv9_GT36-v2/NANOAOD", {sXS: -1}),
    ("/SingleMuon/Run2017F-UL2017_MiniAODv2_NanoAODv9_GT36-v1/NANOAOD", {sXS: -1}),
    #("/SingleMuon/Run2017G-UL2017_MiniAODv2_NanoAODv9_GT36-v2/NANOAOD", {sXs: -1}), ## these are low PU that are not used in most analysis : https://twiki.cern.ch/twiki/bin/viewauth/CMS/LumiRecommendationsRun2#2017
    #("/SingleMuon/Run2017H-UL2017_MiniAODv2_NanoAODv9_GT36-v1/NANOAOD", {sXs: -1}),

    ## EGamma
    # XS (cross-section) does not matter for data sample
    ("/SingleElectron/Run2017B-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD", {sXS: -1}),
    ("/SingleElectron/Run2017C-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD", {sXS: -1}),
    ("/SingleElectron/Run2017D-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD", {sXS: -1}),
    ("/SingleElectron/Run2017E-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD", {sXS: -1}),
    ("/SingleElectron/Run2017F-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD", {sXS: -1})
])
