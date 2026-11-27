"""Refresh published, independently calculated CS2 team Elo ratings."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
import re
import unicodedata

import requests

ANALYZER = "https://cselo.net"
PREDICT = "https://csgopredict.org/rankings/"
FRAGCHARTS = "https://www.fragcharts.com/rankings/elo"
MAX_RATING_AGE = timedelta(days=3)
TIMEOUT = (3, 6)
TEAM_ALIASES = {
    "navi": "natusvincere",
    "nip": "ninjasinpyjamas",
    "teamliquid": "liquid",
    "teamspirit": "spirit",
    "lunacyteam": "lunacy",
    "xiesport": "xi",
}


def team_key(name):
    name = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().lower()
    key = re.sub(r"[^a-z0-9]", "", name)
    return TEAM_ALIASES.get(key, key)


class RankingTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.text = []
        self.row = None
        self.cell = None
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip = True
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = False
        elif tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if not self.skip:
            if self.cell is not None:
                self.cell.append(data)
            self.text.append(data)


def parse_table(html):
    table = RankingTable()
    table.feed(html)
    ratings, ambiguous = {}, set()
    columns = None
    for row in table.rows:
        labels = [cell.casefold() for cell in row]
        if "team" in labels and "elo" in labels:
            columns = labels.index("team"), labels.index("elo")
            continue
        if columns is None or len(row) <= max(columns):
            continue
        key, rating = team_key(row[columns[0]]), re.sub(r"[,\s]", "", row[columns[1]])
        if not key or not rating.isdecimal() or not 100 <= int(rating) <= 4000:
            continue
        if key in ratings:
            ambiguous.add(key)
        ratings[key] = int(rating)
    return {key: value for key, value in ratings.items() if key not in ambiguous}, " ".join(table.text)


def get_page(url, **kwargs):
    response = requests.get(url, timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    if float(response.headers.get("Age", 0)) > 3600:
        raise ValueError("Ranking page was served from an old cache")
    return response.text


@dataclass
class Ratings:
    name: str
    as_of: datetime
    teams: dict
    per_map: bool = False


def fragcharts_ratings(reference):
    page = RankingTable()
    page.feed(get_page(FRAGCHARTS))
    words = [part.strip() for part in page.text if part.strip()]
    updated = re.search(r"Fragcharts rating\s*·\s*(\d{1,2} [A-Za-z]+ \d{4})", " ".join(words), re.I)
    if not updated:
        raise ValueError("Fragcharts did not return a dated ranking")
    months = "January February March April May June July August September October November December".split()
    day, month, year = updated[1].split()
    if month.title() not in months:
        raise ValueError("Fragcharts returned an unknown rating month")
    observed = datetime(int(year), months.index(month.title()) + 1, int(day), tzinfo=timezone.utc)
    if not timedelta(0) <= reference - observed <= MAX_RATING_AGE:
        raise ValueError("Fragcharts ranking is outdated")
    ratings, ambiguous = {}, set()
    for i, value in enumerate(words[1:-2], 1):
        if not re.fullmatch(r"(?:[1-3],\d{3}|[5-9]\d{2})", value):
            continue
        if not re.fullmatch(r"(?:[+\-−]\d+|—)", words[i + 1]):
            continue
        context = " ".join(words[i + 2:i + 6])
        if not re.match(r"(?:Valve\s+#\s*\d|no Valve file)", context, re.I):
            continue
        label = words[i - 1]
        key = team_key(label)
        if not key or label.isdecimal():
            continue
        if key in ratings:
            ambiguous.add(key)
        ratings[key] = int(value.replace(",", ""))
    if len(ratings) < 20:
        raise ValueError("Fragcharts did not return a complete team ranking")
    return Ratings("Fragcharts", observed, {key: value for key, value in ratings.items() if key not in ambiguous}, True)


def analyzer_ratings(reference):
    # The home page identifies the age of the live ratings. Paginated tables
    # carry the full ranking, including teams outside the home-page top 20.
    html = get_page(ANALYZER)
    _, words = parse_table(html)
    updated = re.search(r"Data is up to date\s*:\s*(?:(\d+)\s*(minute|hour|day|second)s?\s+ago|just now)", words, re.I)
    if not updated:
        raise ValueError("CS ELO Analyzer did not publish a rating age")
    unit = updated[2] or "second"
    observed = reference - timedelta(**{unit.lower() + "s": int(updated[1] or 0)})
    ratings = {}
    ambiguous = set()
    for page_number in range(1, 11):
        url = ANALYZER + "/teams"
        page, words = parse_table(get_page(url) if page_number == 1 else get_page(url, params={"page": page_number}))
        if len(page) < 2 and page_number == 1:
            raise ValueError("CS ELO Analyzer did not return a team ranking")
        ambiguous.update(ratings.keys() & page.keys())
        ratings.update(page)
        if not re.search(r"\bNext\b", words):
            break
    if observed > reference or reference - observed > MAX_RATING_AGE:
        raise ValueError("CS ELO Analyzer ranking is outdated")
    return Ratings("CS ELO Analyzer", observed, {key: value for key, value in ratings.items() if key not in ambiguous})


def predict_ratings(reference):
    ratings, words = parse_table(get_page(PREDICT))
    updated = re.search(r"Last updated\s+(\d{4}-\d\d-\d\d\s+\d\d:\d\d)\s+UTC", words, re.I)
    if len(ratings) < 2 or not updated:
        raise ValueError("CSGOPredict did not return a dated ranking")
    observed = datetime.fromisoformat(" ".join(updated[1].split())).replace(tzinfo=timezone.utc)
    if not timedelta(0) <= reference - observed <= MAX_RATING_AGE:
        raise ValueError("CSGOPredict ranking is outdated")
    return Ratings("CSGOPredict", observed, ratings)
