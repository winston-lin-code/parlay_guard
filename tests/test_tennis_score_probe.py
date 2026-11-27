from datetime import timedelta

from src.tennis_score_probe import Comparison, matched_feeds, now


def snapshots(clock, us_points, other_points):
    slug = "itfme-one-two"
    start = (clock - timedelta(minutes=25)).isoformat()
    event = {
        "slug": slug, "title": "One vs. Two", "live": True, "active": True, "closed": False,
        "startTime": start, "score_received_at": clock,
        "eventState": {"type": "tennis", "live": True, "score": f"4-2:{'-'.join(us_points)}",
                       "period": "S1", "updatedAt": (clock - timedelta(seconds=2)).isoformat()},
        "markets": [{"slug": f"aec-{slug}", "marketType": "moneyline",
                     "sportsMarketType": "tennis_match_winner", "active": True, "closed": False,
                     "marketSides": [{"team": {"ordering": "away", "name": "Two"}},
                                     {"team": {"ordering": "home", "name": "One"}}]}],
    }
    other = {
        "id": 42, "tour": "itf", "draw": "singles", "gender": "men", "scheduled_time": start,
        "status": "live", "players": {"p1": {"name": "Two"}, "p2": {"name": "One"}},
        "score": {"timestamp": (clock - timedelta(seconds=25)).isoformat(), "observed_age_seconds": 1,
                  "games": [[2], [4]], "points": other_points[::-1]},
    }
    return {slug: event}, [other], clock


def test_probe_compares_new_score_states_in_same_player_order():
    clock = now()
    probe = Comparison()
    probe.observe(*snapshots(clock, ["15", "0"], ["15", "0"]))
    probe.observe(*snapshots(clock + timedelta(seconds=5), ["15", "0"], ["30", "0"]))
    probe.observe(*snapshots(clock + timedelta(seconds=10), ["30", "0"], ["30", "0"]))
    probe.observe(*snapshots(clock + timedelta(seconds=15), ["30", "0"], ["30", "0"]))
    report = probe.result()
    assert report["matched_live_polls"] == 4
    assert report["us_within_20_seconds"] == 4
    assert report["alternate_older_state_recently_confirmed"] == 4
    assert report["distinct_changes_seen_on_both"] == 1
    assert report["median_alternate_lead_seconds"] == 5


def test_probe_drops_ambiguous_provider_matches():
    events, peers, _ = snapshots(now(), ["15", "0"], ["15", "0"])
    assert matched_feeds(events, [peers[0], {**peers[0], "id": 81}]) == []
