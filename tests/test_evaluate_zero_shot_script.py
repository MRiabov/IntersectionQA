import os

from scripts.evaluate_zero_shot import _load_env_file


def test_load_env_file_preserves_existing_values(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_API_KEY=from_file\n"
        "export HF_TOKEN='hf_from_file'\n"
        'CUSTOM_VALUE="quoted value"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "from_shell")

    _load_env_file(path)

    assert os.environ["OPENAI_API_KEY"] == "from_shell"
    assert os.environ["HF_TOKEN"] == "hf_from_file"
    assert os.environ["CUSTOM_VALUE"] == "quoted value"
