from pathlib import Path
from behavior_study.site_data import build_snapshot

if __name__=='__main__':
    data=build_snapshot(Path(__file__).resolve().parents[1])
    print('Updated site_data/experiment.json. Independent evaluation:',data['independent_evaluation']['status'])
