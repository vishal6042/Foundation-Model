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


def test_kasteren_a(tmp_path):
    from homefm.data.converters.kasteren import load_kasteren

    (tmp_path / "sensorData.txt").write_text(
        "Start time\tEnd time\tID\tVal\n----\t----\t--\t---\n"
        "25-Feb-2008 09:37:51\t25-Feb-2008 09:37:52\t14\t1\n"
        "25-Feb-2008 09:36:43\t25-Feb-2008 09:37:04\t5\t1\n"
        "Length: 2\n"
    )
    (tmp_path / "activitiesData.txt").write_text(
        "Start time\tEnd time\tID\n----\t----\t--\n25-Feb-2008 09:37:17\t25-Feb-2008 09:38:02\t4\n"
    )
    s = load_kasteren(tmp_path, "A")
    assert s.home_id == "kasteren_a" and [t.state for t in s.tokens] == [State.ON, State.OFF, State.ON, State.OFF]
    assert s.tokens[0].entity.item == "hall toilet door" and s.tokens[0].entity.sensor_type == "door contact"
    assert s.tokens[2].entity.room == "toilet" and s.tokens[2].entity.sensor_type == "float"
    assert s.episodes[0].concept == "use toilet" and s.episodes[0].end_ts - s.episodes[0].start_ts == 45


def test_kasteren_c(tmp_path):
    from homefm.data.converters.kasteren import load_kasteren

    (tmp_path / "sensor_labels.txt").write_text("5 BedRight BedroomUpstairs\n25 ToiletDoorDownstairs BathroomDownstairs (Toilet)\n")
    (tmp_path / "activity_labels.txt").write_text("8 TakeBath (NoInstance)\n10 GoToBed\n")
    (tmp_path / "sensors.csv").write_text("19-Nov-2008 22:51:02,19-Nov-2008 22:51:04,25,1\n"
                                          "20-Nov-2008 01:39:31,20-Nov-2008 01:39:46,5,1\n")
    (tmp_path / "activities.csv").write_text("19-Nov-2008 23:00:50,20-Nov-2008 01:44:59,10\n")
    s = load_kasteren(tmp_path, "C")
    rooms = {e.item: (e.room, e.sensor_type) for e in s.entities}
    assert rooms == {"toilet door downstairs": ("bathroom downstairs", "door contact"),
                     "bed right": ("bedroom upstairs", "pressure")}
    assert len(s.tokens) == 4 and s.episodes[0].concept == "go to bed"


def test_mural(tmp_path):
    import json

    from homefm.baselines.domusfm import build_domus_dataset
    from homefm.data.converters.mural import load_mural

    root = tmp_path / "MuRAL"
    (root / "01").mkdir(parents=True)
    (root / "sensors.json").write_text(json.dumps([
        {"sensor name": "bedroom_1 door sensor", "sensor type": "magnetic contact sensor",
         "sensor location": "bedroom_1 door"},
        {"sensor name": "kitchen counter motion sensor", "sensor type": "PIR motion sensor",
         "sensor location": "kitchen"}]))
    (root / "activities.json").write_text(json.dumps([{"id": 0, "name": "others"}, {"id": 2, "name": "cooking"},
                                                      {"id": 6, "name": "leaving bedroom"}]))
    (root / "01" / "context.json").write_text(json.dumps({"start time": "23:59:00", "day": "weekend"}))
    (root / "01" / "data.csv").write_text(
        "uid,time,sensor,action,Subject,Description,activity\n"
        '0,23:59:58,bedroom_1 door,OPENED,A,"A opens, then walks",6\n'
        "1,23:59:58,kitchen counter mov,turned ON,B,B cooks,2\n"
        "2,00:00:03,,,B,B takes a knife,2\n"
        "3,00:00:05,kitchen counter mov,turned OFF,B,B leaves,0\n")
    s = load_mural(tmp_path)
    assert [t.state for t in s.tokens] == [State.ON, State.ON, State.OFF]
    assert abs(s.tokens[1].ts - s.tokens[0].ts - 1e-3) < 1e-6 and s.tokens[2].ts - s.tokens[0].ts == 7  # crosses midnight
    assert [t.person_id for t in s.tokens] == ["A", "B", "B"]
    assert {(e.item, e.room, e.sensor_type) for e in s.entities} == {("door", "bedroom 1", "door contact"),
                                                                       ("counter motion", "kitchen", "motion")}
    d = build_domus_dataset("mural", [s])  # per-event labels despite two residents in the same second
    assert [d.activities[i] for i in d.label] == ["leaving bedroom", "cooking", "Other"]


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
