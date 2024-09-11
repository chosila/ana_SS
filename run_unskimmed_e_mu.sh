python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2018 -run_mode condor -v unskimmed_egamma -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 5 -excludeSamples SingleMuon_Run2018A,SingleMuon_Run2018B,SingleMuon_Run2018C,SingleMuon_Run2018D -leptonSelection Electron -dryRun

python3 htoaa_Wrapper.py -analyze htoaa_triggerStudy_GGFMode.py -era 2018 -run_mode condor -v unskimmed_singlemuon -ntuples UnskimmedHToAATo4BNanoAOD -nFilesPerJob 5 -excludeSamples EGamma_Run2018A,EGamma_Run2018B,EGamma_Run2018C,EGamma_Run2018D -leptonSelection Muon -dryRun
