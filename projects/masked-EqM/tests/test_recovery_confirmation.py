from pathlib import Path
import sys, numpy as np, pytest
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from energy_outcome_monotonicity.aggregate_recovery_confirmation import seed_effect
def test_seed_effect_is_paired_and_positive(tmp_path):
 p=tmp_path/'x.npz'; np.savez(p,dot_error=np.ones((4,2)),direct_error=np.zeros((4,2)))
 mean,ci=seed_effect(p,bootstrap=100,seed=1); assert mean==1 and ci[0]==1
def test_seed_effect_rejects_unpaired_shapes(tmp_path):
 p=tmp_path/'x.npz'; np.savez(p,dot_error=np.ones((4,2)),direct_error=np.ones((3,2)))
 with pytest.raises(ValueError): seed_effect(p,bootstrap=10,seed=1)
