from worst_shinka.cli import integrations


def test_play_candidate_allows_models_from_different_runs(monkeypatch, tmp_path):
    model_run = tmp_path / "run_a"
    opponent_run = tmp_path / "run_b"
    model_path = model_run / "gen_0" / "model.pt"
    opponent_path = opponent_run / "gen_2" / "model.pt"
    model_path.parent.mkdir(parents=True)
    opponent_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    opponent_path.write_bytes(b"opponent")

    captured = {}

    class FakeRLModule:
        @staticmethod
        def run_play(*, model_path=None, opponent_path=None, stop_event=None, **kwargs):
            captured["model_path"] = model_path
            captured["opponent_path"] = opponent_path
            captured["stop_event"] = stop_event

    monkeypatch.setattr(integrations, "configure_run", lambda run_dir: run_dir)
    monkeypatch.setattr(integrations, "_rl_modules", lambda: {"run_play": FakeRLModule})

    integrations.play_candidate(model_path=str(model_path), opponent_path=str(opponent_path), stop_event=None)

    assert captured["model_path"] == str(model_path)
    assert captured["opponent_path"] == str(opponent_path)
    assert captured["stop_event"] is None
