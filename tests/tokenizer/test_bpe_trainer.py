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


def _naive_train(words, num_merges):
    """The original recount-everything trainer, kept as an oracle."""
    from collections import Counter

    from snapdragon.tokenizer.bpe_reference import _merge_word

    words = dict(words)
    merges = []
    for step in range(num_merges):
        pairs = Counter()
        for word, freq in words.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        merges.append(best)
        merged = {}
        for word, freq in words.items():
            key = _merge_word(word, best, 256 + step) if len(word) > 1 else word
            merged[key] = merged.get(key, 0) + freq
        words = merged
    return merges


def test_incremental_trainer_matches_naive_oracle():
    text = [
        "the quick brown fox jumps over the lazy dog " * 4,
        "she sells sea shells by the sea shore " * 6,
        "banana bandana cabana " * 9,
        "numbers 1234 5678 90 and symbols !?! ... ;;",
    ]
    words = count_words(text)
    assert train_merges(words, 80) == _naive_train(words, 80)
