from ontime.config import load_config, project_path


def test_config_loads_and_has_seed():
    cfg = load_config()
    assert cfg["seed"] == 42


def test_split_dates_are_ordered():
    s = load_config()["splits"]
    assert s["train_end"] < s["val_start"] <= s["val_end"] < s["test_start"] < s["censoring_cutoff"]


def test_project_path_is_absolute():
    assert project_path("data/raw").is_absolute()
