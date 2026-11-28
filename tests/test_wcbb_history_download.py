from unittest.mock import MagicMock

import pandas as pd
import pytest

from src import wcbb_history_download as history


def test_builds_membership_only_from_dated_division_i_espn_team_rows():
    groups = pd.DataFrame(
        [("wbb", year, "subdivision", f"wbb:division-{year}", "NCAA Division I", "D-I", "D1")
         for year in (*history.SEASONS, 2027)], columns=history.GROUPS)
    teams = pd.DataFrame(
        [("wbb", year, team, source, group, None)
         for year in (*history.SEASONS, 2027)
         for team, source, group in (
             ("101", "espn", f"wbb:division-{year}"),
             ("102", "espn", "wbb:division-ii"),
             ("103", "ncaa", f"wbb:division-{year}"),
         )], columns=history.TEAMS)
    members = history.membership(groups, teams)
    assert list(members.season) == list(history.SEASONS)
    assert list(members.team_id.unique()) == ["101"]
    with pytest.raises(ValueError, match="one unambiguous"):
        history.membership(groups[groups.season != 2024], teams)


def test_download_validates_before_publishing_and_reuses_completed_file(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    session = MagicMock()
    response = session.return_value.__enter__.return_value.get.return_value.__enter__.return_value
    response.ok = True
    monkeypatch.setattr(history.requests, "Session", session)
    response.iter_content.return_value = [b"<html>Not a Parquet file</html>"]
    with pytest.raises(ValueError):
        history.fetch(tmp_path, "release", "sample", ("season",))
    assert not list(tmp_path.iterdir())

    response.iter_content.return_value = [pd.DataFrame({"season": [2026]}).to_parquet(index=False)]
    path = history.fetch(tmp_path, "release", "sample", ("season",))
    assert list(pd.read_parquet(path).season) == [2026]
    assert history.fetch(tmp_path, "release", "sample", ("season",)) == path
    assert session.call_count == 2
