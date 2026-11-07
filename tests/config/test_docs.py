import importlib.util
from pathlib import Path

from snapdragon.config.docs import render_config_reference

ROOT = Path(__file__).resolve().parents[2]


def test_reference_lists_every_section_and_key():
    text = render_config_reference()
    for heading in ("## `model`", "## `data`", "## `optim`", "## `optim.scheduler`", "## `train`",
                    "## `train.eval`", "## `data.sources`"):
        assert heading in text
    assert "| `n_kv_heads` | `int` | `5` |" in text
    assert "| `precision` | `str` | `'bf16'` |" in text
    assert "| `adam_betas` |" in text


def test_every_dataclass_field_is_documented():
    import dataclasses

    from snapdragon.config.schema import Config

    text = render_config_reference()
    for section in dataclasses.fields(Config):
        for f in dataclasses.fields(getattr(Config(), section.name)):
            assert f"| `{f.name}` |" in text, f"{section.name}.{f.name}"


def test_committed_docs_match_the_code():
    """If this fails: python scripts/gen_config_docs.py"""
    path = ROOT / "docs" / "configuration.md"
    assert path.exists(), "run scripts/gen_config_docs.py"
    assert path.read_text(encoding="utf-8") == render_config_reference()


def test_script_check_mode_exit_codes(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("gen_docs_cli", ROOT / "scripts" / "gen_config_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "TARGET", tmp_path / "configuration.md")
    assert mod.main(["--check"]) == 1  # missing file is out of date
    assert mod.main([]) == 0
    assert mod.main(["--check"]) == 0
