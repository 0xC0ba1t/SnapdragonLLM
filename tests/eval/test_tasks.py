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


ARC_ROW = {
    "id": "x",
    "question": "Which gas do plants absorb?",
    "choices": {"text": ["oxygen", "carbon dioxide", "helium"], "label": ["A", "B", "C"]},
    "answerKey": "B",
}


def test_arc_example_format_and_label_lookup():
    from snapdragon.eval.tasks import arc_example

    ex = arc_example(ARC_ROW)
    assert ex.context == "Question: Which gas do plants absorb?\nAnswer:"
    assert ex.choices == [" oxygen", " carbon dioxide", " helium"]
    assert ex.label == 1


def test_arc_handles_numeric_answer_keys():
    from snapdragon.eval.tasks import arc_example

    row = {**ARC_ROW, "choices": {"text": ["a", "b", "c", "d"], "label": ["1", "2", "3", "4"]}, "answerKey": "4"}
    assert arc_example(row).label == 3


def test_arc_tasks_accept_injected_rows_and_limit():
    from snapdragon.eval.tasks import arc_challenge, arc_easy

    assert len(list(arc_easy(iter([ARC_ROW] * 4)))) == 4
    assert len(list(arc_challenge(iter([ARC_ROW] * 4), limit=2))) == 2
    assert "arc_easy" in TASKS
    assert "arc_challenge" in TASKS
