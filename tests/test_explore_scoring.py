from src.explore.grid import Candidate
from src.explore.scoring import Blacklist, Scored, rank, score


def test_blacklist_radius_and_ttl():
    b = Blacklist(radius_m=1.0, ttl_s=600.0)
    b.add(5.0, 5.0, now=100.0)
    assert b.blocked(5.5, 5.0, now=200.0)
    assert not b.blocked(6.5, 5.0, now=200.0)
    assert not b.blocked(5.0, 5.0, now=701.0)


def test_score_prefers_gain_per_meter():
    assert score(100, 10.0) == 10.0
    assert score(100, 0.1) == 200.0


def test_rank_sorts_and_filters():
    a = Scored(Candidate(0, 0, "frontier"), gain=10, path_len_m=1.0, score=10.0)
    b = Scored(Candidate(1, 0, "frontier"), gain=50, path_len_m=1.0, score=50.0)
    c = Scored(Candidate(2, 0, "under_seen"), gain=1, path_len_m=1.0, score=1.0)
    assert [s.score for s in rank([a, b, c], min_score=5.0)] == [50.0, 10.0]
