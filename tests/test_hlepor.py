import pytest

from hlepor import *
from hlepor import _separate_to_words, _separate_sentences, _count_words, _get_identical_words, _calc_aligned_num, \
    _calc_precision, _calc_recall, _find_position_difference, _calc_hlepor


def test_hlepor_score_with_diff_len_lists():
    reference = ['Test sentence 1.', 'Test sentence 2.']
    hypothesis = ['Test sentence 1.']
    with pytest.raises(ValueError):
        hlepor_score(reference, hypothesis)


def test_separate_to_words_one_sent():
    reference = ['this is a reference line.']
    hypothesis = ['this is a hypothesis line.']
    assert _separate_to_words(reference, hypothesis) == ([['this', 'is', 'a', 'reference', 'line', '.']], np.array([6]),
                                                         [['this', 'is', 'a', 'hypothesis', 'line', '.']],
                                                         np.array([6]))


def test_separate_to_words_list_sent():
    reference = ['This is a reference line1.', 'At least they say so.']
    hypothesis = ['this is a hypothesis line1.', 'They claim at least.']
    t = _separate_to_words(reference, hypothesis)
    assert t[0] == [['this', 'is', 'a', 'reference', 'line1', '.'], ['at', 'least', 'they', 'say', 'so', '.']]
    assert t[2] == [['this', 'is', 'a', 'hypothesis', 'line1', '.'], ['they', 'claim', 'at', 'least', '.']]
    np.testing.assert_array_equal(t[1], np.array([6, 6]))
    np.testing.assert_array_equal(t[3], np.array([6, 5]))
    assert len(t) == 4


def test_separate_sentence():
    sentence = ['this is a line.']
    res = _separate_sentences(sentence)
    assert res[0] == [['this', 'is', 'a', 'line', '.']]
    np.testing.assert_array_equal(res[1], np.array([5]))


def test_separate_sentence_with_comma():
    sentence = ['Eats, shoots and leaves.']
    res = _separate_sentences(sentence)
    assert res[0] == [['eats', ',', 'shoots', 'and', 'leaves', '.']]
    np.testing.assert_array_equal(res[1], np.array([6]))


def test_separate_two_sentences():
    sentences = ['This is a line.', 'At least they say so.']
    res = _separate_sentences(sentences)
    assert res[0] == [['this', 'is', 'a', 'line', '.'], ['at', 'least', 'they', 'say', 'so', '.']]
    np.testing.assert_array_equal(res[1], np.array([5, 6]))


def _separate_sentences_attribute_error():
    sentence = [5, 3]
    with pytest.raises(AttributeError):
        _separate_sentences(sentence)


def test_separate_with_empty_sentence():
    sentences = ['This is a line.', '']
    with pytest.raises(ValueError):
        _separate_sentences(sentences)


def test_enhanced_length_penalty():
    reference_length = np.array([6, 6])
    hypothesis_length = np.array([6, 5])
    np.testing.assert_array_equal(enhanced_length_penalty(reference_length, hypothesis_length),
                                  np.array([1, np.exp(1 - 6 / 5)]))


def test_count_words_with_repeat():
    sentence = ['at', 'least', 'they', 'say', 'so', 'and', 'so', '.']
    d = {'at': 1, 'least': 1, 'they': 1, 'say': 1, 'so': 2, 'and': 1, '.': 1}
    assert _count_words(sentence) == d


def test_count_words():
    sentence = ['at', 'least', 'they', 'say', 'so', '.']
    d = {'at': 1, 'least': 1, 'they': 1, 'say': 1, 'so': 1, '.': 1}
    assert _count_words(sentence) == d


def test_get_identical_words_one_word():
    ref_words_list = ['one']
    hypo_words_list = ['one']
    assert _get_identical_words(ref_words_list, hypo_words_list) == {'one': 1}


def test_get_identical_words_one_word_hypo_more():
    ref_words_list = ['one']
    hypo_words_list = ['one', 'two']
    assert _get_identical_words(ref_words_list, hypo_words_list) == {'one': 1}


def test_get_identical_words_one_word_ref_more():
    ref_words_list = ['one', 'two']
    hypo_words_list = ['one']
    assert _get_identical_words(ref_words_list, hypo_words_list) == {'one': 1}


def test_get_identical_words_many_words_ref_more():
    ref_words_list = ['one', 'two', 'one', 'three', 'three', 'two', 'cat']
    hypo_words_list = ['one', 'one', 'one', 'three', 'two']
    assert _get_identical_words(ref_words_list, hypo_words_list) == {'one': 2, 'two': 1, 'three': 1}


def test_get_identical_words_no_one():
    ref_words_list = ['three', 'three', 'cat']
    hypo_words_list = ['one', 'one', 'one', 'two']
    assert _get_identical_words(ref_words_list, hypo_words_list) == {}


def test_calc_aligned_num_no_one():
    assert _calc_aligned_num({}) == 0


def test_calc_aligned_num_one_word():
    assert _calc_aligned_num({'one': 1}) == 1


def test_calc_aligned_num_one_word2():
    assert _calc_aligned_num({'one': 2}) == 2


def test_calc_aligned_num_many_words():
    assert _calc_aligned_num({'one': 1, 'two': 2}) == 3


def test_calc_precision_one_sentence():
    ref_words_list = ['one', 'two', 'one', 'three', 'three', 'two', 'cat']
    hypo_words_list = ['one', 'one', 'one', 'three', 'two']
    identical_words = _get_identical_words(ref_words_list, hypo_words_list)  # {'one': 2, 'two': 1, 'three': 1}
    aligned_num = _calc_aligned_num(identical_words)
    np.testing.assert_array_equal(_calc_precision(aligned_num, len(hypo_words_list)), np.array([4 / 5]))


def test_calc_precision_many_sentences():
    ref_words_list = [['three', 'one', 'cat'], ['one', 'one', 'two']]
    hypo_words_list = [['one', 'one', 'one', 'two'], ['one', 'one', 'one', 'two']]
    aligned_num = [0] * 2
    identical_words = _get_identical_words(ref_words_list[0], hypo_words_list[0])  # [{'one': 1}, {'one': 2, 'two': 1}]
    aligned_num[0] = _calc_aligned_num(identical_words)
    identical_words = _get_identical_words(ref_words_list[1], hypo_words_list[1])  # [{'one': 1}, {'one': 2, 'two': 1}]
    aligned_num[1] = _calc_aligned_num(identical_words)
    np.testing.assert_array_equal(
        _calc_precision(np.array(aligned_num), np.array([len(hypo_words_list[0]), len(hypo_words_list[1])])),
        np.array([1 / 4, 3 / 4]))


def test_calc_precision_no_matched():
    ref_words_list = ['three', 'three', 'cat']
    hypo_words_list = ['one', 'one', 'one', 'two']
    identical_words = {}
    aligned_num = [0]
    np.testing.assert_array_equal(_calc_precision(np.array(aligned_num), np.array([len(hypo_words_list)])),
                                  np.array(aligned_num))


def test_calc_precision_no_matched_in_one_of_two():
    ref_words_list = [['three', 'three', 'cat'], ['one', 'one', 'two']]
    hypo_words_list = [['one', 'one', 'one', 'two'], ['one', 'one', 'one', 'two']]
    aligned_num = [0] * 2
    identical_words = _get_identical_words(ref_words_list[0], hypo_words_list[0])  # [{}, {'one': 2, 'two': 1}]
    aligned_num[0] = _calc_aligned_num(identical_words)
    identical_words = _get_identical_words(ref_words_list[1], hypo_words_list[1])  # [{'one': 1}, {'one': 2, 'two': 1}]
    aligned_num[1] = _calc_aligned_num(identical_words)
    np.testing.assert_array_equal(
        _calc_precision(np.array(aligned_num), np.array([len(hypo_words_list[0]), len(hypo_words_list[1])])),
        np.array([0 / 4, 3 / 4]))


def test_calc_recall_one_sentence():
    ref_words_list = ['one', 'two', 'one', 'three', 'three', 'two', 'cat']
    hypo_words_list = ['one', 'one', 'one', 'three', 'two']
    identical_words = _get_identical_words(ref_words_list, hypo_words_list)  # {'one': 2, 'two': 1, 'three': 1}
    aligned_num = _calc_aligned_num(identical_words)
    np.testing.assert_array_equal(_calc_precision(aligned_num, np.array([len(ref_words_list)])), np.array([4 / 7]))


def test_calc_recall_many_sentences():
    ref_words_list = [['three', 'one', 'cat'], ['one', 'one', 'two']]
    hypo_words_list = [['one', 'one', 'one', 'two'], ['one', 'one', 'one', 'two']]
    aligned_num = [0] * 2
    identical_words = _get_identical_words(ref_words_list[0], hypo_words_list[0])  # [{'one': 1}, {'one': 2, 'two': 1}]
    aligned_num[0] = _calc_aligned_num(identical_words)
    identical_words = _get_identical_words(ref_words_list[1], hypo_words_list[1])  # [{'one': 1}, {'one': 2, 'two': 1}]
    aligned_num[1] = _calc_aligned_num(identical_words)
    np.testing.assert_array_equal(
        _calc_recall(np.array(aligned_num), np.array([len(ref_words_list[0]), len(ref_words_list[1])])),
        np.array([1 / 3, 3 / 3]))


def test_calc_recall_no_matched():
    ref_words_list = ['three', 'three', 'cat']
    hypo_words_list = ['one', 'one', 'one', 'two']
    identical_words = {}
    aligned_num = [0]
    np.testing.assert_array_equal(_calc_precision(np.array(aligned_num), np.array([len(ref_words_list)])),
                                  np.array(aligned_num))


def test_calc_recall_no_matched_in_one_of_two():
    ref_words_list = [['three', 'three', 'cat'], ['one', 'one', 'two']]
    hypo_words_list = [['one', 'one', 'one', 'two'], ['one', 'one', 'one', 'two']]
    aligned_num = [0] * 2
    identical_words = _get_identical_words(ref_words_list[0], hypo_words_list[0])  # [{}, {'one': 2, 'two': 1}]
    aligned_num[0] = _calc_aligned_num(identical_words)
    identical_words = _get_identical_words(ref_words_list[1], hypo_words_list[1])  # [{'one': 1}, {'one': 2, 'two': 1}]
    aligned_num[1] = _calc_aligned_num(identical_words)
    np.testing.assert_array_equal(
        _calc_precision(np.array(aligned_num), np.array([len(ref_words_list[0]), len(ref_words_list[1])])),
        np.array([0 / 3, 3 / 3]))


def test_enhanced_length_penalty():
    reference = ['At least they say so.']
    hypothesis = ['They claim at least.']
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    assert np.sum(np.abs(enhanced_length_penalty(ref_len, hypo_len) - np.array([np.exp(-1 / 5)]))) < 0.000000001


def test_calc_harmonic_mean_P_R():
    reference = ['At least they say so.']
    hypothesis = ['They claim at least.']
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    aligned_num = _calc_aligned_num(_get_identical_words(ref[0], hypo[0]))
    assert np.sum(np.abs(calc_harmonic_mean_p_r(aligned_num, ref_len, hypo_len) - np.array([40 / 59]))) < 0.000000001


def test_find_position_difference():
    reference = ['At least they say so.']
    hypothesis = ['They claim at least.']
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    identical_words_dict = _get_identical_words(ref[0], hypo[0])
    assert np.abs(_find_position_difference(ref[0], hypo[0], ref_len[0], hypo_len[0],
                                            identical_words_dict) - 18 / 15) < 0.000000001


def test_enhanced_length_penalty_equal_repeats():
    reference = ['At least they say and say so.']
    hypothesis = ['They claim at least and say so.']
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    assert np.sum(np.abs(enhanced_length_penalty(ref_len, hypo_len) - np.array([1]))) < 0.000000001


def test_calc_harmonic_mean_p_r_equal_repeats():
    reference = ['At least they say and say so.']
    hypothesis = ['They claim at least and say so.']
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    aligned_num = _calc_aligned_num(_get_identical_words(ref[0], hypo[0]))
    assert np.sum(np.abs(calc_harmonic_mean_p_r(aligned_num, ref_len, hypo_len) - np.array([7 / 8]))) < 0.000000001


def test_find_position_difference_equal_repeats():
    reference = ['At least they say and say so.']
    hypothesis = ['They claim at least and say so.']
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    identical_words_dict = _get_identical_words(ref[0], hypo[0])
    assert np.abs(
        _find_position_difference(ref[0], hypo[0], ref_len[0], hypo_len[0], identical_words_dict) - 3 / 4) < 0.000000001


def test_enhanced_length_penalty_change_pos_repeats():
    reference = ['At least they say and say so.']
    hypothesis = ['They say so and claim at least.']
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    assert np.sum(np.abs(enhanced_length_penalty(ref_len, hypo_len) - np.array([1]))) < 0.000000001


def test_calc_harmonic_mean_p_r_change_pos_repeats():
    reference = ['At least they say and say so.']
    hypothesis = ['They say so and claim at least.']
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    aligned_num = _calc_aligned_num(_get_identical_words(ref[0], hypo[0]))
    assert np.sum(np.abs(calc_harmonic_mean_p_r(aligned_num, ref_len, hypo_len) - np.array([7 / 8]))) < 0.000000001


def test_find_position_difference_change_pos_repeats():
    reference = ['At least they say and say so.']
    hypothesis = ['They say so and claim at least.']
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    identical_words_dict = _get_identical_words(ref[0], hypo[0])
    assert np.abs(_find_position_difference(ref[0], hypo[0], ref_len[0], hypo_len[0],
                                            identical_words_dict) - 19 / 8) < 0.000000001


def test_enhanced_length_penalty_h_more_than_r_repeats():
    reference = ['At least they say so.']
    hypothesis = ['They say and say so and claim at least.']
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    assert np.sum(np.abs(enhanced_length_penalty(ref_len, hypo_len) - np.array([np.exp(-2 / 3)]))) < 0.000000001


def test_calc_harmonic_mean_p_r_h_more_than_r_repeats():
    reference = ['At least they say so.']
    hypothesis = ['They say and say so and claim at least.']
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    aligned_num = _calc_aligned_num(_get_identical_words(ref[0], hypo[0]))
    assert np.sum(np.abs(calc_harmonic_mean_p_r(aligned_num, ref_len, hypo_len) - np.array([15 / 16]))) < 0.000000001


def test_find_position_difference_h_more_than_r_repeats():
    reference = ['At least they say so.']
    hypothesis = ['They say and say so and claim at least.']
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    identical_words_dict = _get_identical_words(ref[0], hypo[0])
    assert np.abs(_find_position_difference(ref[0], hypo[0], ref_len[0], hypo_len[0],
                                            identical_words_dict) - 36 / 15) < 0.000000001


def test_hlepor_score():
    reference = ['At least they say so.']
    hypothesis = ['They claim at least.']
    assert np.abs(hlepor_score(reference, hypothesis) - 10 / (
                2 / np.exp(-1 / 5) + 1 / np.exp(-6 / 25) + 7 * 59 / 40)) < 0.000000005


def test_hlepor_h_more_than_r_score():
    reference = ['At least they say so.']
    hypothesis = ['They say and say so and claim at least.']
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    identical_words_dict = _get_identical_words(ref[0], hypo[0])
    aligned_num = _calc_aligned_num(identical_words_dict)

    lp = enhanced_length_penalty(ref_len, hypo_len)
    pd = _find_position_difference(ref[0], hypo[0], ref_len[0], hypo_len[0], identical_words_dict)
    npd = 1 / hypo_len * pd
    hpr = calc_harmonic_mean_p_r(aligned_num, ref_len, hypo_len)
    assert np.abs(hlepor_score(reference, hypothesis) -
                  10 / (2 / lp + 1 / np.exp(-npd) + 7 / hpr)) < 0.0000005


def test_hlepor_two_sentences_score():
    reference = ['At least they say so.', 'At least they say so.']
    hypothesis = ['They claim at least.', 'They say and say so and claim at least.']
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    identical_words_dict = _get_identical_words(ref[1], hypo[1])
    aligned_num = _calc_aligned_num(identical_words_dict)
    lp = enhanced_length_penalty(ref_len, hypo_len)
    pos_dif = _find_position_difference(ref[1], hypo[1], ref_len[1], hypo_len[1], identical_words_dict)
    npd = 1 / hypo_len[1] * pos_dif
    hpr = calc_harmonic_mean_p_r(aligned_num, ref_len, hypo_len)
    assert np.abs(hlepor_score(reference, hypothesis)
                  - np.mean([10 / (2 / np.exp(-1 / 5) + 1 / np.exp(-6 / 25) + 7 * 59 / 40),
                             10 / (2 / lp[1] + 1 / np.exp(-npd) + 7 / hpr[1])])) < 0.00000005


def test_hlepor_no_match_score():
    reference = ['At least they say so.', 'At least they say so.']
    hypothesis = ['This is empty sentence', 'Eat, shoot and leaves']
    assert np.abs(hlepor_score(reference, hypothesis, separate_punctuation=False) - 0) < 0.00000005


def test_hlepor_equal_score():
    reference = ['At least they say so.', 'Eat, shoot and leaves']
    hypothesis = ['At least they say so.', 'Eat, shoot and leaves']
    assert np.abs(hlepor_score(reference, hypothesis) - 1) < 0.00000005


def test_hlepor_none_score():
    reference = [34, 56]
    hypothesis = [45, 35]
    ref, ref_len, hypo, hypo_len = _separate_to_words(reference, hypothesis)
    assert hlepor_score(reference, hypothesis) is None


def test_single_hlepor_score():
    reference = 'this is a cat'
    hypothesis = 'non matching hypothesis'
    assert np.abs(single_hlepor_score(reference, hypothesis) - 0) < 0.00000005


def test_hlepor_score_example():
    hypothesis = ['It is a guide to action which ensures that the military always obeys the commands of the party',
                  'It is to insure the troops forever hearing the activity guidebook that party direct']
    reference = ['It is a guide to action that ensures that the military will forever heed Party commands',
                 'It is the practical guide for the army always to heed the directions of the party']
    assert np.abs(hlepor_score(reference, hypothesis) - 0.6214) < 0.00005


def test_single_hlepor_score2():
        hypothesis = 'It is a guide to action which ensures that the military always obeys the commands of the party'
        reference = 'It is a guide to action that ensures that the military will forever heed Party commands'
        assert np.abs(single_hlepor_score(reference, hypothesis) - 0.7842) < 0.0005
