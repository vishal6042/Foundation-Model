from homefm.data.converters.casas_zenodo import load_casas_zenodo, parse_sensor_name
from homefm.data.converters.uci_adl import load_uci_adl
from homefm.schema import State


def test_parse_sensor_name():
    assert parse_sensor_name("KitchenAStove") == ("stove", "kitchen")
    assert parse_sensor_name("BedroomBArea") == ("second bedroom area", "second bedroom")
    assert parse_sensor_name("BedroomABedB") == ("bed", "bedroom")
    assert parse_sensor_name("LoungeChair") == ("lounge chair", "living room")
    assert parse_sensor_name("Bathroom") == ("bathroom area", "bathroom")


def test_casas_zenodo(tmp_path):
    f = tmp_path / "hh999.csv"
    f.write_text(
        "2012-07-20,10:00:00.0,Kitchen,ON,Cook=\"begin\"\n"
        "2012-07-20,10:00:05.0,Kitchen,OFF\n"
        "2012-07-20,10:01:00.0,OutsideDoor,OPEN\n"
        "2012-07-20,10:02:00.0,KitchenATemperature,21\n"
        "2012-07-20,10:10:00.0,Kitchen,ON,Cook=\"end\"\n"
        "2012-07-20,10:20:00.0,Bathroom,ON,Toilet\n"
    )
    s = load_casas_zenodo(f)
    assert s.home_id == "hh999" and len(s.tokens) == 6
    types = {e.entity_id: e.sensor_type for e in s.entities}
    assert types["OutsideDoor|door contact"] == "door contact"
    assert types["KitchenATemperature|temperature"] == "temperature"
    assert s.tokens[2].state == State.ON
    eps = {(e.concept, e.end_ts - e.start_ts) for e in s.episodes}
    assert ("cook", 600.0) in eps and ("toilet", 0.0) in eps


def test_uci_adl(tmp_path):
    (tmp_path / "OrdonezB_Sensors.txt").write_text(
        "Start time\tEnd time\tLocation\tType\tPlace\n----\t----\t----\t----\t----\n"
        "2012-11-11 21:14:21\t\t2012-11-12 00:21:49\t\tSeat\t\tPressure\tLiving\n"
    )
    (tmp_path / "OrdonezB_ADLs.txt").write_text(
        "Start time\tEnd time\tActivity\n----\t----\t----\n"
        "2012-11-11 21:14:00\t\t2012-11-12 00:22:59\t\tSpare_Time/TV\n"
    )
    s = load_uci_adl(tmp_path, "B")
    assert [t.state for t in s.tokens] == [State.ON, State.OFF]
    assert s.entities[0].sensor_type == "pressure" and s.entities[0].room == "living"
    assert s.episodes[0].concept == "spare time/tv"


def test_casas_fast_matches_reference(tmp_path):
    import numpy as np

    from homefm.baselines.domusfm import build_domus_dataset, build_domus_from_arrays
    from homefm.data.converters.casas_fast import load_casas_fast

    f = tmp_path / "hh998.csv"
    f.write_text(
        "2012-07-20,10:00:00.0,Kitchen,ON,Cook=\"begin\"\n"
        "2012-07-20,10:00:05.0,Kitchen,OFF\n"
        "2012-07-20,10:01:00.0,OutsideDoor,OPEN\n"
        "2012-07-20,10:02:00.0,KitchenATemperature,21\n"
        "2012-07-20,10:03:00.0,KitchenATemperature,26\n"
        "2012-07-20,10:10:00.0,Kitchen,ON,Cook=\"end\"\n"
    )
    a = build_domus_from_arrays(load_casas_fast(f, cache_dir=tmp_path / "cache"))
    a2 = build_domus_from_arrays(load_casas_fast(f, cache_dir=tmp_path / "cache"))  # from cache
    b = build_domus_dataset("hh998", [load_casas_zenodo(f)])
    for x in (a, a2):
        assert x.n_events == b.n_events
        assert (x.status == b.status).all()
        assert list(np.array(x.activities)[x.label]) == list(np.array(b.activities)[b.label])


def test_balanced_sampler():
    import numpy as np

    from homefm.baselines.domusfm.train import BalancedSampler

    idx = np.array(list(BalancedSampler([10, 1000], 4000)))
    assert idx.min() >= 0 and idx.max() < 1010
    small = (idx < 10).mean()
    assert 0.4 < small < 0.6  # each dataset drawn about equally often
