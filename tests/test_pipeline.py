"""Check entry points protect existing sensing data and select fresh outputs."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def driver(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("pipeline_under_test", PROJECT/"scripts/run_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROJECT", tmp_path)
    return module


def invoke(driver, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", *map(str, args)])
    driver.main()


def test_sensing_preflight_protects_entire_batch(driver, monkeypatch, tmp_path):
    jobs = [("first", {"seed": 1}), ("second", {"seed": 2})]
    monkeypatch.setattr(driver, "showcase_configs", lambda: iter(jobs))
    batch = tmp_path/"batch"
    saved = batch/"second"
    saved.mkdir(parents=True)
    artifact = saved/"visibility_continuous_input.json"
    artifact.write_text('{"historical": true}')
    before = artifact.read_bytes()
    with pytest.raises(ValueError, match="existing sensing results preserved"):
        invoke(driver, monkeypatch, "--showcase", "--output", batch, "--stage", "all")
    assert artifact.read_bytes() == before
    assert not (batch/"first").exists()
    assert not (tmp_path/"configs").exists()


def test_showcase_respects_saved_config_and_fresh_output(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "showcase_configs", lambda: iter([("first", {"seed": 1})]))
    config = tmp_path/"configs/showcase/first.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"seed": 99}\n')
    original = config.read_bytes()
    generated = []
    monkeypatch.setattr(driver, "generate", lambda cfg, folder: generated.append((cfg, folder)) or
                        {"room_count_achieved": 7, "obstacle_count_achieved": 0,
                         "accepted_attempt": 1, "generation_seconds": 0})
    monkeypatch.setattr(driver, "collect", lambda *args: None)
    batch = tmp_path/"fresh"
    invoke(driver, monkeypatch, "--showcase", "--output", batch, "--stage", "generate")
    assert generated == [({"seed": 99}, batch/"first")]
    assert config.read_bytes() == original


def test_unknown_showcase_selection_fails_before_writing(driver, monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        invoke(driver, monkeypatch, "--showcase", "--only", "missing", "--stage", "generate")
    assert list(tmp_path.iterdir()) == []


def test_default_output_stays_inside_outputs(driver, monkeypatch, tmp_path):
    config=tmp_path/"config.json"
    config.write_text('{"seed": 1}')
    generated=[]
    monkeypatch.setattr(driver, "generate", lambda cfg, folder: generated.append(folder) or
                        {"room_count_achieved": 1, "obstacle_count_achieved": 0,
                         "accepted_attempt": 0, "generation_seconds": 0})
    invoke(driver, monkeypatch, "--config", config, "--stage", "generate")
    assert generated == [tmp_path/"outputs/example"]
