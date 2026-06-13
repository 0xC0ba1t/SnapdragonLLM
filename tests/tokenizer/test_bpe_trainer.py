from snapdragon.tokenizer.bpe_reference import count_words, num_merges_for, train_merges


def test_most_frequent_pair_merges_first():
    words = count_words(["aaab aaab aaab"])
    merges = train_merges(words, 1)
    assert merges == [(ord("a"), ord("a"))]


def test_training_is_deterministic():
    text = ["the quick brown fox jumps over the lazy dog " * 5, "low lower lowest " * 7]
    a = train_merges(count_words(text), 30)
    b = train_merges(count_words(text), 30)
    assert a == b
    assert len(a) == 30


def test_stops_when_nothing_left_to_merge():
    merges = train_merges(count_words(["ab"]), 50)
    assert len(merges) == 1  # only one pair exists, then the word is a single token


def test_merge_budget():
    assert num_merges_for(32768) == 32768 - 256 - 7
