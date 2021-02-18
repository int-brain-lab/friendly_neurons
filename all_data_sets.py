
from ibl_pipeline import subject, acquisition, ephys, histology
from ibl_pipeline.analyses import behavior as behavior_analysis

print((acquisition.Session & (behavior_analysis.SessionTrainingStatus & 'good_enough_for_brainwide_map=1')).fetch('session_uuid'))