import os
import numpy as np
from collections import OrderedDict as OD

sXRange = "xAxisRange"; sYRange = "yAxisRange";
sXLabel = 'xAxisLabel'; sYLabel = 'yAxisLabel';
sNRebinX = 'nRebinX';  sNRebinY = 'nRebinY';
sXRebinning = 'xRebinning'; sYRebinning = 'yRebinning';

ExpData_dict = {
    'Data ABCD': ['SingleMuon_Run2018A', 'SingleMuon_Run2018B', 'SingleMuon_Run2018C', 'SingleMuon_Run2018D'],
    # 'Data ABCD': ['EGamma_Run2018A', 'EGamma_Run2018B', 'EGamma_Run2018C', 'EGamma_Run2018D'],
    #'Data A': ['SingleMuon_Run2018A'],
    #'Data B': ['SingleMuon_Run2018B'],
    #'Data C': ['SingleMuon_Run2018C'],
    #'Data D': ['SingleMuon_Run2018D']
}
MCBkg_list = [
    'QCD', #'QCD_0bCat', 'QCD_1bCat', 'QCD_2bCat', 'QCD_3bCat', 'QCD_4bCat', 'QCD_5bAndMoreCat',
    # 'TTToHadronic_powheg', 'TTToSemiLeptonic_powheg', 'TTTo2L2Nu_powheg', .
    'TTToHadronic_powheg',
    'TTToSemiLeptonic_powheg_bbqq',  'TTToSemiLeptonic_powheg_bbq' ,'TTToSemiLeptonic_powheg_bb', 'TTToSemiLeptonic_powheg_bqq', 'TTToSemiLeptonic_powheg_1b', 'TTToSemiLeptonic_powheg_0b',
    'TTTo2L2Nu_powheg_bb',        'TTTo2L2Nu_powheg_1b',        'TTTo2L2Nu_powheg_0b',
    "SingleTop",
    #'ZJetsToQQ_HT',
    #"DYJets_M-10to50_Incl_LO", "DYJets_M-50_Incl_LO",
    "DYJets_M-10to50_Incl_NLO", "DYJets_M-50_Incl_NLO",
    #"DYJets_M-10to50_Incl_LO",
    # "DYJets_HT_LO",
    # "DYJets_M-50_Incl_NLO",
    #'WJetsToQQ_HT',
    # "DYJets_M-50_HT_LO",
    'WJetsToLNu_HT_LO', #'WJetsToLNu_Incl_NLO', #'WJetsToLNu_HT_LO', #'WJetsToLNu_Incl_NLO',
]

MCSig_list = [] #['SUSY_GluGluH_01J_HToAATo4B_M-20_HPtAbv150', 'SUSY_GluGluH_01J_HToAATo4B_M-30_HPtAbv150', ]
sLableSig = [] #['HToAATo4B_M-20', 'HToAATo4B_M-30']
systematics_list = ['central']
systematics_forData = 'noweight'
selectionTags = ['sel_1b', 'sel_0b', 'sel_0b_BBQ','sel_0b_BBQQ']#['sel_JetID', 'sel_Mass140','sel_Mass140_dR_2p75']#['sel_JetID', 'sel_leadingFatJetZHbb_Xbb_avg', 'SR'] #['SR', 'SR_Trg_Combo_AK4AK8Jet_HT'] #['SR', 'SR_TrgAK8330_M30_BDBnp4', 'SR_Trg2AK4116_DCSVp71', 'SR_TrgAK8400_M30', 'SR_TrgAK8500', 'SR_TrgComb2', 'SR_TrgComb4' ] # ['sel_JetID', 'sel_lFJPNetXbbPlusDZHbb', 'sel_L1_SingleJet180', 'SR'] # ['sel_JetID', 'sel_L1_SingleJet180', 'SR'] #['sel_HLT_IsoMu27'] # ['sel_JetID', 'sel_L1_SingleJet180', 'SR'] # ['sel_JetID', 'sel_L1_SingleJet180', 'SR'] # ['sel_leadingFatJetMSoftDrop', 'sel_leadingFatJetParticleNetMD_XbbvsQCD', 'SR'] #['SR', 'sel_leadingFatJetMSoftDrop', 'sel_leadingFatJetParticleNetMD_XbbvsQCD', 'sel_2018HEM1516Issue']

HLT_toUse = 'HLT_IsoMu24'

scale_MCSig = 50 #1000
yRatioLimit = [0.4, 1.6]

logYMinScaleFactor = 1 # 10 # 100 # 1 # scale yMin by factor logYMinScaleFactor to not concentrate lowest stats background processes

VariableBinning_pT = np.array([*np.arange(0,300,20), *np.arange(300,600,50), *np.arange(600,1001,100)])

histograms_dict = OD([
    ("hLeadingMuonPt", {sXLabel: 'hLeadingMuonPt', sYLabel: 'Events', sXRange: [1, 500], sNRebinX: 8 }),
    ("hLeadingMuonEta", {sXLabel: 'hLeadingMuonEta', sYLabel: 'Events', sXRange: [-3.5, 3.5], sNRebinX: 2 }),
    ("hLeadingMuonPhi", {sXLabel: 'hLeadingMuonPhi', sYLabel: 'Events', sXRange: [-3.14, 3.14], sNRebinX: 2 }),

    ("hLeadingElectronEta", {sXLabel: 'hLeadingElectronEta', sYLabel: 'Events', sXRange: [-3.5, 3.5], sNRebinX: 2 }),
    ("hLeadingElectronPhi", {sXLabel: 'hLeadingElectronPhi', sYLabel: 'Events', sXRange: [-3.14, 3.14], sNRebinX: 2 }),
    ("hdR_leadingMuon_leadingFatJet", {sXLabel: 'hdR_leadingMuon_leadingFatJet', sYLabel: 'Events',  sNRebinX: 10 }),
    ("hLeadingFatJetPt", {sXLabel: 'hLeadingFatJetPt', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 8 }),
    ("hLeadingFatJetPt", {sXLabel: 'hLeadingFatJetPt', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 8 }),
    ("hLeadingFatJetEta", {sXLabel: 'hLeadingFatJetEta', sYLabel: 'Events', sXRange: [-3.5, 3.5], sNRebinX: 2 }),
    ("hLeadingFatJetPhi", {sXLabel: 'hLeadingFatJetPhi', sYLabel: 'Events', sXRange: [-3.14, 3.14], sNRebinX: 2 }),
    ("hLeadingFatJetMass", {sXLabel: 'hLeadingFatJetMass', sYLabel: 'Events', sXRange: [1, 300], sNRebinX: 10}),
    ("hLeadingFatJetMSoftDrop", {sXLabel: 'hLeadingFatJetMSoftDrop', sYLabel: 'Events', sXRange: [1, 300], sNRebinX: 10 }),
    ("hLeadingFatJetParticleNetMD_XbbOverQCD", {sXLabel: 'hLeadingFatJetParticleNetMD_XbbOverQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 4 }),
    ("hLeadingFatJet_nBHadrons", {sXLabel: 'nBHadrons', sYLabel: 'Events', sXRange: [0,6], sNRebinX:1}),
    ("nBQuarkFromTop", {sXLabel: 'nBQuarkFromTop', sYLabel: 'Events', sXRange: [0,6], sNRebinX:1}),
    ("nLightQuarkFromTop", {sXLabel: 'nLightQuarkFromTop', sYLabel: 'Events', sXRange: [0,6], sNRebinX:1}),
    ('bdt_mass_fat', {sXLabel: 'bdt_mass_fat', sYLabel: 'Events', sXRange: [1,400], sNRebinX:8}),
    ('flavB_max_jet', {sXLabel: 'flavB_max_jet', sYLabel: 'Events', sXRange: [0,1], sNRebinX:4}),
    ('mass_lvJ', {sXLabel: 'mass_lvJ', sYLabel: 'Events', sXRange: [1,1000], sNRebinX:10}),
    ('dR_lep_fat', {sXLabel: 'dR_lep_fat', sYLabel: 'Events', sXRange: [0,5], sNRebinX:10}),
    ('flavB_near_lJ', {sXLabel: 'flavB_near_lJ', sYLabel: 'Events', sXRange: [0,1], sNRebinX: 4}), # mlscore axis
    ('pt_jet1', {sXLabel: 'pt_jet1', sYLabel: 'Events', sXRange: [1,1500], sNRebinX:20}), # pt axis
    ('dEta_lep_fat', {sXLabel: 'dEta_lep_fat', sYLabel: 'Events', sXRange: [-3.5, 3.5], sNRebinX:2}), # eta axis
    ('pt_jet3', {sXLabel: 'pt_jet3', sYLabel: 'Events', sXRange: [1,1500], sNRebinX:5}), # pt axis
    ('dPhi_lv_fat', {sXLabel: 'dPhi_lv_fat', sYLabel: 'Events', sXRange: [-3.14, 3.14], sNRebinX:2}), # phi axis
    ('dR_fat_jet_min', {sXLabel: 'dR_fat_jet_min', sYLabel: 'Events', sXRange: [0,5], sNRebinX:10}), # delta r axis
    ('flavB_near_lep', {sXLabel: 'flavB_near_lep', sYLabel: 'Events', sXRange: [0,1], sNRebinX: 4}),
    ('jet_pt_near_lep', {sXLabel: 'jet_pt_near_lep', sYLabel: 'Events', sXRange: [1,1500], sNRebinX: 20}),
    ('mass_lJ', {sXLabel: 'mass_lJ', sYLabel: 'Events', sXRange: [0, 1000], sNRebinX: 10}),
    ('dR_lJ_flavB_max', {sXLabel: 'dR_lJ_flavB_max', sYLabel: 'Events', sXRange: [0,5], sNRebinX: 10}),
    ('pt_ISRs', {sXLabel: 'pt_ISRs', sYLabel: 'Events', sXRange: [1,1500], sNRebinX: 20}),
    ('bbqq4_xgb_score', {sXLabel: 'bbqq4_xgb_score', sYLabel: 'Events', sXRange: [0,1], sNRebinX:4}),
    ('bbqq11_xgb_score', {sXLabel: 'bbqq11_xgb_score', sYLabel: 'Events', sXRange: [0,1], sNRebinX:4}),
    ('bbqq_bbq13_xgb_score', {sXLabel: 'bbqq_bbq13_xgb_score', sYLabel: 'Events', sXRange: [0,1], sNRebinX:4}),
    ('FatJet_PNetMD_Hto4b_Htoaa4bOverQCD', {sXLabel: 'FatJet_PNetMD_Hto4b_Htoaa4bOverQCD', sYLabel: 'Events', sXRange: [0,1], sNRebinX:4}),
    ('FatJet_PNetMD_Hto4b_Htoaa3bOverQCD', {sXLabel: 'FatJet_PNetMD_Hto4b_Htoaa3bOverQCD', sYLabel: 'Events', sXRange: [0,1], sNRebinX:4}),
    ('btagHbb', {sXLabel: 'btagHbb', sYLabel: 'Events', sXRange: [-1,1], sNRebinX:4}),
    ('particleNetMD_Xbb', {sXLabel: 'particleNetMD_Xbb', sYLabel: 'Events', sXRange: [0,1], sNRebinX:4}),
    ('btagDDBvLV2', {sXLabel: 'btagDDBvLV2', sYLabel: 'Events', sXRange: [0,1], sNRebinX:4}),
    ('deepTagMD_ZHbbvsQCD', {sXLabel: 'deepTagMD_ZHbbvsQCD', sYLabel: 'Events', sXRange: [0,1], sNRebinX:4}),
    ("hLeadingFatJetParticleNetMD_XbbOverQCD", {sXLabel: 'hLeadingFatJetParticleNetMD_XbbOverQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),

])
