import os
import numpy as np
from collections import OrderedDict as OD

sXRange = "xAxisRange"; sYRange = "yAxisRange";
sXLabel = 'xAxisLabel'; sYLabel = 'yAxisLabel';
sNRebinX = 'nRebinX';  sNRebinY = 'nRebinY'; 
sXRebinning = 'xRebinning'; sYRebinning = 'yRebinning'; 

ExpData_dict = {
    'Data ABCD': ['JetHT_Run2018A', 'JetHT_Run2018B', 'JetHT_Run2018C', 'JetHT_Run2018D'],
    #'Data A': ['JetHT_Run2018A'],
    #'Data B': ['JetHT_Run2018B'],
    #'Data C': ['JetHT_Run2018C'],
    #'Data D': ['JetHT_Run2018D']
}
MCBkg_list = ['QCD_0bCat', 'QCD_1bCat', 'QCD_2bCat', 'QCD_3bCat', 'QCD_4bCat', 'QCD_5bAndMoreCat',  'TTToHadronic_powheg', 'TTToSemiLeptonic_powheg', 'TTTo2L2Nu_powheg', 'WJetsToQQ_HT', 'ZJetsToQQ_HT']
#MCBkg_list = ['QCD_0bCat', 'QCD_1bCat', 'QCD_2bCat', 'QCD_3bCat', 'QCD_4bCat', 'QCD_5bAndMoreCat',  ]
MCSig_list = [
    #'SUSY_GluGluH_01J_HToAATo4B_M-15_HPtAbv150', 
    #'SUSY_GluGluH_01J_HToAATo4B_M-20_HPtAbv150', 
    #'SUSY_GluGluH_01J_HToAATo4B_M-25_HPtAbv150', 
    #'SUSY_GluGluH_01J_HToAATo4B_M-30_HPtAbv150', 
    #'SUSY_GluGluH_01J_HToAATo4B_M-50_HPtAbv150', 
    'SUSY_GluGluH_01J_HToAATo4B_M-55_HPtAbv150', 
    ]
sLableSig = [
    #'HToAATo4B_M-15', 
    #'HToAATo4B_M-20', 
    #'HToAATo4B_M-25', 
    #'HToAATo4B_M-30', 
    #'HToAATo4B_M-50',
    'HToAATo4B_M-55',
     ]
systematics_list = ['central']
systematics_forData = 'noweight'
selectionTags = ['SRWP40_mA55Window'] # ['SRWP40'] # ['sel_leadingFatJetMSoftDrop', 'sel_leadingFatJetParticleNetMD_XbbvsQCD', 'SR'] #['SR', 'sel_leadingFatJetMSoftDrop', 'sel_leadingFatJetParticleNetMD_XbbvsQCD', 'sel_2018HEM1516Issue']

#HLT_toUse = 'HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4'
HLT_toUse = 'Trg_Combo_AK4AK8Jet_HT'

scale_MCSig = 1 #2 #50 #1000
yRatioLimit = [0.4, 1.6]

logYMinScaleFactor = 10 # 100 # 1 # scale yMin by factor logYMinScaleFactor to not concentrate lowest stats background processes



histograms_dict = OD([
    #("hLeadingFatJetMass", {sXLabel: 'Leading FatJet mass [GeV]', sYLabel: 'Events', sXRange: [50, 250], sYRange: [1e-2, 1e8]})
    #("hLeadingFatJetMass", {sXLabel: 'Leading FatJet mass [GeV]', sYLabel: 'Events', sXRange: [50, 250]}),
    
    #("", {sXLabel: '', sYLabel: 'Events', sXRange: []}),
    
    #("hCutFlow", {sXLabel: 'hCutFlow', sYLabel: 'Events'}),
    #("hCutFlowWeighted", {sXLabel: 'hCutFlowWeighted', sYLabel: 'Events'}),

    #("hPV_npvs_beforeSel", {sXLabel: 'No. of primary vertices before selection', sYLabel: 'Events', sXRange: [0, 100] }),
    #("hPV_npvsGood_beforeSel", {sXLabel: 'No. of good primary vertices before selection', sYLabel: 'Events', sXRange: [0, 100] }),
    #("hPV_npvs_SR", {sXLabel: 'No. of primary vertices in SR', sYLabel: 'Events', sXRange: [0, 100] }),
    #("hPV_npvsGood_SR", {sXLabel: 'No. of good primary vertices in SR', sYLabel: 'Events', sXRange: [0, 100] }),

    ("hLeadingFatJetPt", {sXLabel: 'hLeadingFatJetPt', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 4 }),
    ("hLeadingFatJetEta", {sXLabel: 'hLeadingFatJetEta', sYLabel: 'Events', sXRange: [-3.5, 3.5], sNRebinX: 2 }),
    ("hLeadingFatJetPhi", {sXLabel: 'hLeadingFatJetPhi', sYLabel: 'Events', sXRange: [-3.14, 3.14], sNRebinX: 2 }),

    #("hLeadingFatJetPt_HEM1516IssueEtaPhiCut", {sXLabel: 'hLeadingFatJetPt_HEM1516IssueEtaPhiCut', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 4 }),
    #("hLeadingFatJetEta_HEM1516IssuePhiCut", {sXLabel: 'hLeadingFatJetEta_HEM1516IssuePhiCut', sYLabel: 'Events', sXRange: [-3.5, 3.5] }),
    #("hLeadingFatJetPhi_HEM1516IssueEtaCut", {sXLabel: 'hLeadingFatJetPhi_HEM1516IssueEtaCut', sYLabel: 'Events', sXRange: [-3.14, 3.14], sNRebinX: 2 }),
    #("hLeadingFatJetPt_HEM1516IssueEtaPhiCut_DataPreHEM1516Issue", {sXLabel: 'hLeadingFatJetPt_HEM1516IssueEtaPhiCut_DataPreHEM1516Issue', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 4 }),
    #("hLeadingFatJetEta_HEM1516IssuePhiCut_DataPreHEM1516Issue", {sXLabel: 'hLeadingFatJetEta_HEM1516IssuePhiCut_DataPreHEM1516Issue', sYLabel: 'Events', sXRange: [-3.5, 3.5] }),
    #("hLeadingFatJetPhi_HEM1516IssueEtaCut_DataPreHEM1516Issue", {sXLabel: 'hLeadingFatJetPhi_HEM1516IssueEtaCut_DataPreHEM1516Issue', sYLabel: 'Events', sXRange: [-3.14, 3.14], sNRebinX: 2 }),
    #("hLeadingFatJetPt_HEM1516IssueEtaPhiCut_DataWithHEM1516Issue", {sXLabel: 'hLeadingFatJetPt_HEM1516IssueEtaPhiCut_DataWithHEM1516Issue', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 4 }),
    #("hLeadingFatJetEta_HEM1516IssuePhiCut_DataWithHEM1516Issue", {sXLabel: 'hLeadingFatJetEta_HEM1516IssuePhiCut_DataWithHEM1516Issue', sYLabel: 'Events', sXRange: [-3.5, 3.5] }),
    #("hLeadingFatJetPhi_HEM1516IssueEtaCut_DataWithHEM1516Issue", {sXLabel: 'hLeadingFatJetPhi_HEM1516IssueEtaCut_DataWithHEM1516Issue', sYLabel: 'Events', sXRange: [-3.14, 3.14], sNRebinX: 2 }),

    #("hLeadingFatJetPt_HEM1516IssueEtaPhiCut_woHEM1516Fix", {sXLabel: 'hLeadingFatJetPt_HEM1516IssueEtaPhiCut_woHEM1516Fix', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 4  }),
    #("hLeadingFatJetEta_HEM1516IssuePhiCut_woHEM1516Fix", {sXLabel: 'hLeadingFatJetEta_HEM1516IssuePhiCut_woHEM1516Fix', sYLabel: 'Events', sXRange: [-3.5, 3.5] }),
    #("hLeadingFatJetPhi_HEM1516IssueEtaCut_woHEM1516Fix", {sXLabel: 'hLeadingFatJetPhi_HEM1516IssueEtaCut_woHEM1516Fix', sYLabel: 'Events', sXRange: [-3.14, 3.14], sNRebinX: 2 }),
    #("hLeadingFatJetPt_HEM1516IssueEtaPhiCut_woHEM1516Fix_DataPreHEM1516Issue", {sXLabel: 'hLeadingFatJetPt_HEM1516IssueEtaPhiCut_woHEM1516Fix_DataPreHEM1516Issue', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 4  }),
    #("hLeadingFatJetEta_HEM1516IssuePhiCut_woHEM1516Fix_DataPreHEM1516Issue", {sXLabel: 'hLeadingFatJetEta_HEM1516IssuePhiCut_woHEM1516Fix_DataPreHEM1516Issue', sYLabel: 'Events', sXRange: [-3.5, 3.5] }),
    #("hLeadingFatJetPhi_HEM1516IssueEtaCut_woHEM1516Fix_DataPreHEM1516Issue", {sXLabel: 'hLeadingFatJetPhi_HEM1516IssueEtaCut_woHEM1516Fix_DataPreHEM1516Issue', sYLabel: 'Events', sXRange: [-3.14, 3.14], sNRebinX: 2 }),
    #("hLeadingFatJetPt_HEM1516IssueEtaPhiCut_woHEM1516Fix_DataWithHEM1516Issue", {sXLabel: 'hLeadingFatJetPt_HEM1516IssueEtaPhiCut_woHEM1516Fix_DataWithHEM1516Issue', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 4 }),
    #("hLeadingFatJetEta_HEM1516IssuePhiCut_woHEM1516Fix_DataWithHEM1516Issue", {sXLabel: 'hLeadingFatJetEta_HEM1516IssuePhiCut_woHEM1516Fix_DataWithHEM1516Issue', sYLabel: 'Events', sXRange: [-3.5, 3.5] }),
    #("hLeadingFatJetPhi_HEM1516IssueEtaCut_woHEM1516Fix_DataWithHEM1516Issue", {sXLabel: 'hLeadingFatJetPhi_HEM1516IssueEtaCut_woHEM1516Fix_DataWithHEM1516Issue', sYLabel: 'Events', sXRange: [-3.14, 3.14], sNRebinX: 2 }),

    # 2018 HEM15/16 issue validation
    #("hLeadingFatJetPt_HEM1516IssueEtaPhiCut_woHEM1516MCRewgt", {sXLabel: 'hLeadingFatJetPt_HEM1516IssueEtaPhiCut_woHEM1516MCRewgt', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 4 }),
    #("hLeadingFatJetEta_HEM1516IssuePhiCut_woHEM1516MCRewgt", {sXLabel: 'hLeadingFatJetEta_HEM1516IssuePhiCut_woHEM1516MCRewgt', sYLabel: 'Events', sXRange: [-3.5, 3.5] }),
    #("hLeadingFatJetPhi_HEM1516IssueEtaCut_woHEM1516MCRewgt", {sXLabel: 'hLeadingFatJetPhi_HEM1516IssueEtaCut_woHEM1516MCRewgt', sYLabel: 'Events', sXRange: [-3.14, 3.14], sNRebinX: 2 }),
    #("hLeadingFatJetPt_HEM1516IssueEtaPhiCut_woHEM1516MCRewgt_DataPreHEM1516Issue", {sXLabel: 'hLeadingFatJetPt_HEM1516IssueEtaPhiCut_woHEM1516MCRewgt_DataPreHEM1516Issue', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 4 }),
    #("hLeadingFatJetEta_HEM1516IssuePhiCut_woHEM1516MCRewgt_DataPreHEM1516Issue", {sXLabel: 'hLeadingFatJetEta_HEM1516IssuePhiCut_woHEM1516MCRewgt_DataPreHEM1516Issue', sYLabel: 'Events', sXRange: [-3.5, 3.5] }),
    #("hLeadingFatJetPhi_HEM1516IssueEtaCut_woHEM1516MCRewgt_DataPreHEM1516Issue", {sXLabel: 'hLeadingFatJetPhi_HEM1516IssueEtaCut_woHEM1516MCRewgt_DataPreHEM1516Issue', sYLabel: 'Events', sXRange: [-3.14, 3.14], sNRebinX: 2 }),
    #("hLeadingFatJetPt_HEM1516IssueEtaPhiCut_woHEM1516MCRewgt_DataWithHEM1516Issue", {sXLabel: 'hLeadingFatJetPt_HEM1516IssueEtaPhiCut_woHEM1516MCRewgt_DataWithHEM1516Issue', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 4 }),
    #("hLeadingFatJetEta_HEM1516IssuePhiCut_woHEM1516MCRewgt_DataWithHEM1516Issue", {sXLabel: 'hLeadingFatJetEta_HEM1516IssuePhiCut_woHEM1516MCRewgt_DataWithHEM1516Issue', sYLabel: 'Events', sXRange: [-3.5, 3.5] }),
    #("hLeadingFatJetPhi_HEM1516IssueEtaCut_woHEM1516MCRewgt_DataWithHEM1516Issue", {sXLabel: 'hLeadingFatJetPhi_HEM1516IssueEtaCut_woHEM1516MCRewgt_DataWithHEM1516Issue', sYLabel: 'Events', sXRange: [-3.14, 3.14], sNRebinX: 2 }),

    ("hLeadingFatJetMass", {sXLabel: 'hLeadingFatJetMass', sYLabel: 'Events', sXRange: [0, 300], sNRebinX: 5}),
    ("hLeadingFatJetMSoftDrop", {sXLabel: 'hLeadingFatJetMSoftDrop', sYLabel: 'Events', sXRange: [0, 300], sNRebinX: 5 }),
    ("hLeadingFatJetBtagDeepB", {sXLabel: 'hLeadingFatJetBtagDeepB', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetBtagDDBvLV2", {sXLabel: 'hLeadingFatJetBtagDDBvLV2', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetBtagDDCvBV2", {sXLabel: 'hLeadingFatJetBtagDDCvBV2', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    
    ("hLeadingFatJetBtagHbb", {sXLabel: 'hLeadingFatJetBtagHbb', sYLabel: 'Events', sXRange: [-1, 1], sNRebinX: 2 }),
    ("hLeadingFatJetDeepTagMD_H4qvsQCD", {sXLabel: 'hLeadingFatJetDeepTagMD_H4qvsQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetDeepTagMD_HbbvsQCD", {sXLabel: 'hLeadingFatJetDeepTagMD_HbbvsQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetDeepTagMD_ZHbbvsQCD", {sXLabel: 'hLeadingFatJetDeepTagMD_ZHbbvsQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetDeepTagMD_ZHccvsQCD", {sXLabel: 'hLeadingFatJetDeepTagMD_ZHccvsQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    
    ("hLeadingFatJetDeepTagMD_ZbbvsQCD", {sXLabel: 'hLeadingFatJetDeepTagMD_ZbbvsQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetDeepTagMD_ZvsQCD", {sXLabel: 'hLeadingFatJetDeepTagMD_ZvsQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetDeepTagMD_bbvsLight", {sXLabel: 'hLeadingFatJetDeepTagMD_bbvsLight', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetDeepTagMD_ccvsLight", {sXLabel: 'hLeadingFatJetDeepTagMD_ccvsLight', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetDeepTag_H", {sXLabel: 'hLeadingFatJetDeepTag_H', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    
    ("hLeadingFatJetDeepTag_QCD", {sXLabel: 'hLeadingFatJetDeepTag_QCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetDeepTag_QCDothers", {sXLabel: 'hLeadingFatJetDeepTag_QCDothers', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetN2b1", {sXLabel: 'hLeadingFatJetN2b1', sYLabel: 'Events', sXRange: [0, 0.6], sNRebinX: 2 }),
    ("hLeadingFatJetN3b1", {sXLabel: 'hLeadingFatJetN3b1', sYLabel: 'Events', sXRange: [0.5, 3.5], sNRebinX: 2 }),
    ("hLeadingFatJetTau1", {sXLabel: 'hLeadingFatJetTau1', sYLabel: 'Events', sXRange: [0, 0.6], sNRebinX: 2 }),    
    ("hLeadingFatJetTau2", {sXLabel: 'hLeadingFatJetTau2', sYLabel: 'Events', sXRange: [0, 0.5], sNRebinX: 2 }),
    ("hLeadingFatJetTau3", {sXLabel: 'hLeadingFatJetTau3', sYLabel: 'Events', sXRange: [0, 0.3], sNRebinX: 2 }),
    ("hLeadingFatJetTau4", {sXLabel: 'hLeadingFatJetTau4', sYLabel: 'Events', sXRange: [0, 0.4], sNRebinX: 2 }),
    ("hLeadingFatJetTau4by3", {sXLabel: 'hLeadingFatJetTau4by3', sYLabel: 'Events', sXRange: [0.2, 1], sNRebinX: 2 }),
    ("hLeadingFatJetTau3by2", {sXLabel: 'hLeadingFatJetTau3by2', sYLabel: 'Events', sXRange: [0.2, 1], sNRebinX: 2 }),    
    ("hLeadingFatJetTau2by1", {sXLabel: 'hLeadingFatJetTau2by1', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),

    ("hLeadingFatJetNConstituents", {sXLabel: 'hLeadingFatJetNConstituents', sYLabel: 'Events'}),
    ("hLeadingFatJetNBHadrons", {sXLabel: 'hLeadingFatJetNBHadrons', sYLabel: 'Events', sXRange: [-0.5, 10.5]}),
    ("hLeadingFatJetNCHadrons", {sXLabel: 'hLeadingFatJetNCHadrons', sYLabel: 'Events', sXRange: [-0.5, 10.5]}),

    ("hLeadingFatJetParticleNetMD_QCD", {sXLabel: 'hLeadingFatJetParticleNetMD_QCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetParticleNetMD_Xbb", {sXLabel: 'hLeadingFatJetParticleNetMD_Xbb', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetParticleNetMD_Xcc", {sXLabel: 'hLeadingFatJetParticleNetMD_Xcc', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),       
    ("hLeadingFatJetParticleNetMD_Xqq", {sXLabel: 'hLeadingFatJetParticleNetMD_Xqq', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2}),

    ("hLeadingFatJetParticleNetMD_XbbOverQCD", {sXLabel: 'hLeadingFatJetParticleNetMD_XbbOverQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }), 
    ("hLeadingFatJetParticleNetMD_XccOverQCD", {sXLabel: 'hLeadingFatJetParticleNetMD_XccOverQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }), 
    ("hLeadingFatJetParticleNetMD_XqqOverQCD", {sXLabel: 'hLeadingFatJetParticleNetMD_XqqOverQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }), 

    ("hLeadingFatJetParticleNet_H4qvsQCD", {sXLabel: 'hLeadingFatJetParticleNet_H4qvsQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetParticleNet_HbbvsQCD", {sXLabel: 'hLeadingFatJetParticleNet_HbbvsQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetParticleNet_HccvsQCD", {sXLabel: 'hLeadingFatJetParticleNet_HccvsQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    ("hLeadingFatJetParticleNet_QCD", {sXLabel: 'hLeadingFatJetParticleNet_QCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: 2 }),
    
    ("hLeadingFatJetParticleNet_mass", {sXLabel: 'hLeadingFatJetParticleNet_mass', sYLabel: 'Events', sXRange: [0, 300], sNRebinX: 5 }),

    ("hLeadingFatJet_nSubJets", {sXLabel: 'hLeadingFatJet_nSubJets', sYLabel: 'Events', sXRange: [-0.5, 6.5] }),
    ("hLeadingFatJet_nSubJets_bTag_L", {sXLabel: 'hLeadingFatJet_nSubJets_bTag_L', sYLabel: 'Events', sXRange: [-0.5, 10.5] }),
    ("hLeadingFatJet_nSubJets_bTag_M", {sXLabel: 'hLeadingFatJet_nSubJets_bTag_M', sYLabel: 'Events', sXRange: [-0.5, 10.5] }),
    ("hLeadingFatJet_nSV", {sXLabel: 'hLeadingFatJet_nSV', sYLabel: 'Events', sXRange: [-0.5, 10.5] }),

    ("hMET_pT", {sXLabel: 'hMET_pT', sYLabel: 'Events', sXRange: [0, 1000], sNRebinX: 5 }),
    ("hMET_sumEt", {sXLabel: 'hMET_sumEt', sYLabel: 'Events', sXRange: [1000, 4000], sNRebinX: 5 }),

    ("hLeadingFatJet_nLeptons", {sXLabel: 'hLeadingFatJet_nLeptons', sYLabel: 'Events', sXRange: [-0.5, 6.5] }),

    #("", {sXLabel: '', sYLabel: 'Events'}),
])

'''
histograms_dict = OD([
    ("hLeadingFatJetPt", {sXLabel: 'hLeadingFatJetPt', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 4 }),
    #("hLeadingFatJetPt", {sXLabel: 'hLeadingFatJetPt', sYLabel: 'Events', sXRange: [800, 900], sYRange: [100, 1000] }),
])
'''

'''
histograms_dict = OD([
    ("hLeadingFatJetEta_vs_Phi", {sXLabel: 'hLeadingFatJetEta', sYLabel: 'hLeadingFatJetPhi', sNRebinX: 2, sNRebinY: 2  }),
    #("hLeadingFatJetPt", {sXLabel: 'hLeadingFatJetPt', sYLabel: 'Events', sXRange: [800, 900], sYRange: [100, 1000] }),
])
'''


nRebinXTmp_ = 50
histograms_dict.update( OD([
    ("hLeadingFatJetParticleNetMD_Hto4b_Haa01b", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_Haa01b', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_Haa2b", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_Haa2b', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_Haa3b", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_Haa3b', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_Haa4b", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_Haa4b', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_QCD0b", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_QCD0b', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_QCD1b", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_QCD1b', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_QCD2b", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_QCD2b', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_QCD3b", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_QCD3b', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_QCD4b", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_QCD4b', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_binaryLF_Haa4b", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_binaryLF_Haa4b', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    
    ("hLeadingFatJetParticleNetMD_Hto4b_binaryLF_QCDlf", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_binaryLF_QCDlf', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_binary_Haa4b", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_binary_Haa4b', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_binary_QCD", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_binary_QCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_Haa34b", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_Haa34b', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_binary_Haa4b_avg", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_binary_Haa4b_avg', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_Haa4b_avg", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_Haa4b_avg', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_QCD01234b", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_QCD01234b', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_binary_QCD_avg", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_binary_QCD_avg', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_QCD_avg", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_QCD_avg', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    
    ("hLeadingFatJetParticleNetMD_Hto4b_Htoaa4bOverQCD", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_Htoaa4bOverQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_Htoaa34bOverQCD", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_Htoaa34bOverQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_binaryLF_Htoaa4bOverQCD", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_binaryLF_Htoaa4bOverQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_binary_Htoaa4bOverQCD", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_binary_Htoaa4bOverQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_binary_Htoaa4bOverQCD_avg", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_binary_Htoaa4bOverQCD_avg', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    ("hLeadingFatJetParticleNetMD_Hto4b_Htoaa4bOverQCD_avg", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_Htoaa4bOverQCD_avg', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),

    ("hLeadingFatJetParticleNet_massA_Hto4b_v0", {sXLabel: 'hLeadingFatJetParticleNet_massA_Hto4b_v0', sYLabel: 'Events', sXRange: [0, 80], sNRebinX: 2}),
    ("hLeadingFatJetParticleNet_massA_Hto4b_v1", {sXLabel: 'hLeadingFatJetParticleNet_massA_Hto4b_v1', sYLabel: 'Events', sXRange: [0, 80], sNRebinX: 2}),
    ("hLeadingFatJetParticleNet_massA_Hto4b_v2", {sXLabel: 'hLeadingFatJetParticleNet_massA_Hto4b_v2', sYLabel: 'Events', sXRange: [0, 80], sNRebinX: 2}),
    ("hLeadingFatJetParticleNet_massA_Hto4b_v3", {sXLabel: 'hLeadingFatJetParticleNet_massA_Hto4b_v3', sYLabel: 'Events', sXRange: [0, 80], sNRebinX: 2}),
    ("hLeadingFatJetParticleNet_massA_Hto4b_v4", {sXLabel: 'hLeadingFatJetParticleNet_massA_Hto4b_v4', sYLabel: 'Events', sXRange: [0, 80], sNRebinX: 2}),
    ("hLeadingFatJetParticleNet_massH_Hto4b_v0", {sXLabel: 'hLeadingFatJetParticleNet_massH_Hto4b_v0', sYLabel: 'Events', sXRange: [0, 300], sNRebinX: 5}),
    ("hLeadingFatJetParticleNet_massH_Hto4b_v00", {sXLabel: 'hLeadingFatJetParticleNet_massH_Hto4b_v00', sYLabel: 'Events', sXRange: [50, 300], sNRebinX: 5}),
    ("hLeadingFatJetParticleNet_massH_Hto4b_v1", {sXLabel: 'hLeadingFatJetParticleNet_massH_Hto4b_v1', sYLabel: 'Events', sXRange: [50, 300], sNRebinX: 5}),
    ("hLeadingFatJetParticleNet_massH_Hto4b_v2", {sXLabel: 'hLeadingFatJetParticleNet_massH_Hto4b_v2', sYLabel: 'Events', sXRange: [50, 300], sNRebinX: 5}),
    ("hLeadingFatJetParticleNet_massH_Hto4b_v3", {sXLabel: 'hLeadingFatJetParticleNet_massH_Hto4b_v3', sYLabel: 'Events', sXRange: [50, 300], sNRebinX: 5}),
    ("hLeadingFatJetParticleNet_massH_Hto4b_v4", {sXLabel: 'hLeadingFatJetParticleNet_massH_Hto4b_v4', sYLabel: 'Events', sXRange: [50, 300], sNRebinX: 5}),
    
    ("hLeadingFatJetParticleNet_massA_Hto4b_avg_v013", {sXLabel: 'hLeadingFatJetParticleNet_massA_Hto4b_avg_v013', sYLabel: 'Events', sXRange: [0, 80], sNRebinX: 2}),    
    ("hLeadingFatJetParticleNet_massH_Hto4b_avg_v0123", {sXLabel: 'hLeadingFatJetParticleNet_massH_Hto4b_avg_v0123', sYLabel: 'Events', sXRange: [50, 300], sNRebinX: 5}),
    
]) )

yRange_tmp_ = [9999,-9999] # [0.0001, 75] # [9999,-9999]# [0.0001, 15]
histograms_dict  = OD([
#    ("hLeadingFatJetParticleNetMD_Hto4b_Haa4b", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_Haa4b', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),    
#    ("hLeadingFatJetMSoftDrop", {sXLabel: 'hLeadingFatJetMSoftDrop', sYLabel: 'Events', sXRange: [0, 300], sNRebinX: 5 }),
#    ("hLeadingFatJetPt", {sXLabel: 'hLeadingFatJetPt', sYLabel: 'Events', sXRange: [180, 1000], sNRebinX: 4 }),
#    ("hLeadingFatJetNBHadrons", {sXLabel: 'hLeadingFatJetNBHadrons', sYLabel: 'Events', sXRange: [-0.5, 10.5]}),
#    ("hLeadingFatJetParticleNetMD_Hto4b_Htoaa4bOverQCD", {sXLabel: 'hLeadingFatJetParticleNetMD_Hto4b_Htoaa4bOverQCD', sYLabel: 'Events', sXRange: [0, 1], sNRebinX: nRebinXTmp_ }),
    
    ("hLeadingFatJetMass", {sXLabel: 'hLeadingFatJetMass', sYLabel: 'Events', sXRange: [0, 300], sNRebinX: 5, sYRange: yRange_tmp_}),
    ("hLeadingFatJetMSoftDrop", {sXLabel: 'hLeadingFatJetMSoftDrop', sYLabel: 'Events', sXRange: [0, 300], sNRebinX: 5, sYRange: yRange_tmp_ }),
    ("hLeadingFatJetParticleNet_mass", {sXLabel: 'hLeadingFatJetParticleNet_mass', sYLabel: 'Events', sXRange: [0, 300], sNRebinX: 5, sYRange: yRange_tmp_ }),
    ("hLeadingFatJetParticleNet_massA_Hto4b_avg_v013", {sXLabel: 'hLeadingFatJetParticleNet_massA_Hto4b_avg_v013', sYLabel: 'Events', sXRange: [0, 80], sNRebinX: 2, sYRange: yRange_tmp_}),    
    ("hLeadingFatJetParticleNet_massH_Hto4b_avg_v0123", {sXLabel: 'hLeadingFatJetParticleNet_massH_Hto4b_avg_v0123', sYLabel: 'Events', sXRange: [50, 300], sNRebinX: 5, sYRange: yRange_tmp_}),
    
])