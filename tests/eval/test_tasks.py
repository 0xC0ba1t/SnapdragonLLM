import json

from snapdragon.eval.tasks import TASKS, hellaswag, hellaswag_example, hellaswag_preprocess

ROW = {
    "activity_label": "Roof shingle removal",
    "ctx_a": "A man is sitting on a roof.",
    "ctx_b": "he",
    "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.",
                "is holding a rubik's cube.", "starts pulling up roofing on a roof."],
    "label": "3",
}


def test_preprocess_cleans_wikihow_artefacts():
    assert hellaswag_preprocess("  Step one [title] do this [header] now  ") == "Step one. do this now"
    assert hellaswag_preprocess("a  b") == "a b"


def test_example_follows_the_harness_format():
    ex = hellaswag_example(ROW)
    assert ex.context == "Roof shingle removal: A man is sitting on a roof. He"
    assert ex.choices[1] == " is ripping level tiles off."
    assert ex.label == 3  # string labels in the file become ints
    assert len(ex.choices) == 4


def test_loader_reads_jsonl_and_honours_limit(tmp_path):
    path = tmp_path / "hs.jsonl"
    path.write_text("\n".join(json.dumps({**ROW, "label": str(i % 4)}) for i in range(5)) + "\n")
    assert len(list(hellaswag(path))) == 5
    assert [e.label for e in hellaswag(path, limit=3)] == [0, 1, 2]


def test_registered_under_its_name():
    assert "hellaswag" in TASKS
    assert TASKS.get("hellaswag") is hellaswag
