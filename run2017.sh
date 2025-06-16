conda activate ana_SS
# electron
# python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2017 -run_mode condor -v unskimmed_egamma_1b_BBQ_BBQQ -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleMuon_Run2017B,SingleMuon_Run2017C,SingleMuon_Run2017D,SingleMuon_Run2017E,SingleMuon_Run2017F -leptonSelection Electron -xgbCut noBdt

#python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2017 -run_mode condor -v unskimmed_EGamma_bdtHi -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleMuon_Run2017B,SingleMuon_Run2017C,SingleMuon_Run2017D,SingleMuon_Run2017E,SingleMuon_Run2017F -leptonSelection Electron -xgbCut bdtHi

#python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2017 -run_mode condor -v unskimmed_EGamma_bdtMed -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleMuon_Run2017B,SingleMuon_Run2017C,SingleMuon_Run2017D,SingleMuon_Run2017E,SingleMuon_Run2017F -leptonSelection Electron -xgbCut bdtMed

#python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2017 -run_mode condor -v unskimmed_EGamma_bdtLo -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleMuon_Run2017B,SingleMuon_Run2017C,SingleMuon_Run2017D,SingleMuon_Run2017E,SingleMuon_Run2017F -leptonSelection Electron -xgbCut bdtLo

#python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2017 -run_mode condor -v unskimmed_EGamma_bdtVeto -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleMuon_Run2017B,SingleMuon_Run2017C,SingleMuon_Run2017D,SingleMuon_Run2017E,SingleMuon_Run2017F -leptonSelection Electron -xgbCut bdtVeto

# muon
# python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2017 -run_mode condor -v unskimmed_singlemuon_1b_BBQ_BBQQ -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleElectron_Run2017B,SingleElectron_Run2017C,SingleElectron_Run2017D,SingleElectron_Run2017E,SingleElectron_Run2017F -leptonSelection Muon -xgbCut noBdt

#python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2017 -run_mode condor -v unskimmed_singlemuon_bdtHi -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleElectron_Run2017B,SingleElectron_Run2017C,SingleElectron_Run2017D,SingleElectron_Run2017E,SingleElectron_Run2017F -leptonSelection Muon -xgbCut bdtHi

#python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2017 -run_mode condor -v unskimmed_singlemuon_bdtMed -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleElectron_Run2017B,SingleElectron_Run2017C,SingleElectron_Run2017D,SingleElectron_Run2017E,SingleElectron_Run2017F -leptonSelection Muon -xgbCut bdtMed

python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2017 -run_mode condor -v unskimmed_singlemuon_bdtLo -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleElectron_Run2017B,SingleElectron_Run2017C,SingleElectron_Run2017D,SingleElectron_Run2017E,SingleElectron_Run2017F -leptonSelection Muon -xgbCut bdtLo

python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2017 -run_mode condor -v unskimmed_singlemuon_bdtVeto -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleElectron_Run2017B,SingleElectron_Run2017C,SingleElectron_Run2017D,SingleElectron_Run2017E,SingleElectron_Run2017F -leptonSelection Muon -xgbCut bdtVeto
