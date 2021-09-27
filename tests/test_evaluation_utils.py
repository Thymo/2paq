from utils.evaluation_utils import exact_match_score_normalized, exact_match_amount, ems, metric_max_over_ground_truths


def test_exact_match_score_normalized():
    assert exact_match_score_normalized('2021', '2021') is True
    assert exact_match_score_normalized('December 2, 2021', '2021') is True
    assert exact_match_score_normalized('December 2, 2021', '2020') is False
    assert exact_match_score_normalized('December 2, 2021', 'December 2021') is True
    assert exact_match_score_normalized('2021', 'December 2021') is False
    assert exact_match_score_normalized('October 2021', 'October 2021') is True


def test_exact_match_amount():
    assert exact_match_amount('25', '250') is False
    assert exact_match_amount('twenty five', '250') is False
    assert exact_match_amount('twenty five', '25') is True
    assert exact_match_amount('1852', '1,852') is True
    assert exact_match_amount('500 years', '500') is True
    assert exact_match_amount('twenty-two', 'twenty two') is True
    assert exact_match_amount('18 chapters', '18') is True
    assert exact_match_amount('8', 'eight') is True
    assert exact_match_amount('607 islands and islets', '607') is True
    assert exact_match_amount('fifth title', '5') is True
    assert exact_match_amount('2,700', 'about 2,700') is True
    assert exact_match_amount('1,800', '1,800 acres') is True
    assert exact_match_amount('five times', '5') is True
    assert exact_match_amount('2 Titles', 'two') is True
    assert exact_match_amount('test', 'test test') is False

    # TODO 6 million becomes -> 6 1000000
    # assert exact_match_amount('over 6 million', 'over six million') is True


def test_metric_max_over_ground_truths():
    assert metric_max_over_ground_truths(ems, 'Los Angeles Dodgers', ['Los Angeles Dodgers'], "") is True
