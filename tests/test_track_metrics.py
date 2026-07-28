import pandas as pd
from metrics.track_metrics import track_balanced_accuracy

def test_long_track_does_not_dominate_track_accuracy():
    records=pd.DataFrame({"group_id":["a"]*100+["b"],"correct":[True]*100+[False]})
    metric, per_track=track_balanced_accuracy(records)
    assert metric==.5 and len(per_track)==2
