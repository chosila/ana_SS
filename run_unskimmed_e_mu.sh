
## older version
#python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2018 -run_mode condor -v unskimmed_egamma -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleMuon_Run2018A,SingleMuon_Run2018B,SingleMuon_Run2018C,SingleMuon_Run2018D -leptonSelection Electron -dryRun

#python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2018 -run_mode condor -v unskimmed_singlemuon -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples EGamma_Run2018A,EGamma_Run2018B,EGamma_Run2018C,EGamma_Run2018D -leptonSelection Muon -dryRun


## muon with xgb cut
# python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2018 -run_mode condor -v unskimmed_singlemuon_bdtHi -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples EGamma_Run2018A,EGamma_Run2018B,EGamma_Run2018C,EGamma_Run2018D -leptonSelection Muon -xgbCut bdtHi

# python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2018 -run_mode condor -v unskimmed_singlemuon_bdtMed -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples EGamma_Run2018A,EGamma_Run2018B,EGamma_Run2018C,EGamma_Run2018D -leptonSelection Muon -xgbCut bdtMed

# python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2018 -run_mode condor -v unskimmed_singlemuon_bdtLo -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples EGamma_Run2018A,EGamma_Run2018B,EGamma_Run2018C,EGamma_Run2018D -leptonSelection Muon -xgbCut bdtLo

# python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2018 -run_mode condor -v unskimmed_singlemuon_bdtVeto -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples EGamma_Run2018A,EGamma_Run2018B,EGamma_Run2018C,EGamma_Run2018D -leptonSelection Muon -xgbCut bdtVeto

## electron with xgb cut
python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2018 -run_mode condor -v unskimmed_EGamma_bdtHi -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleMuon_Run2018A,SingleMuon_Run2018B,SingleMuon_Run2018C,SingleMuon_Run2018D -leptonSelection Electron -xgbCut bdtHi

# python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2018 -run_mode condor -v unskimmed_EGamma_bdtMed -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleMuon_Run2018A,SingleMuon_Run2018B,SingleMuon_Run2018C,SingleMuon_Run2018D -leptonSelection Electron -xgbCut bdtMed
#
# python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2018 -run_mode condor -v unskimmed_EGamma_bdtLo -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleMuon_Run2018A,SingleMuon_Run2018B,SingleMuon_Run2018C,SingleMuon_Run2018D -leptonSelection Electron -xgbCut bdtLo
#
# python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2018 -run_mode condor -v unskimmed_EGamma_bdtVeto -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 2 -excludeSamples SingleMuon_Run2018A,SingleMuon_Run2018B,SingleMuon_Run2018C,SingleMuon_Run2018D -leptonSelection Electron -xgbCut bdtVeto
