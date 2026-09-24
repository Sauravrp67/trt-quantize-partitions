from harness.power import summarize_power

def test_sumarize_power_basic():
    s = summarize_power([10.0,20.0,30.0],dt_s = 0.5)
    assert s.n == 3
    assert s.mean_w == 20.0
    assert s.peak_w == 30.0
    assert abs(s.energy_j - 30.0) < 1e-9

def test_summarize_power_empty():
    s = summarize_power([],dt_s = 0.1)
    assert s.n == 0 and s.mean_w == 0.0 and s.peak_w == 0.0 and s.energy_j == 0.0