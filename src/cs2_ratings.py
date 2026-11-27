"""Refresh published, independently calculated CS2 team Elo ratings."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import unicodedata

import requests

ANALYZER = "https://cselo.net"
PREDICT = "https://csgopredict.org/rankings/"
FRAGCHARTS = "https://www.fragcharts.com/rankings/elo"
MAX_RATING_AGE = timedelta(days=3)
TIMEOUT = (3, 6)
ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "models" / "cs2_elo_seed.json"
CACHE_PATH = ROOT / "data" / "processed" / "cs2_elo.json"
SOURCE_ORDER = ("Fragcharts", "CS ELO Analyzer", "CSGOPredict")
TEAM_ALIASES = {
    "navi": "natusvincere",
    "nip": "ninjasinpyjamas",
    "teamliquid": "liquid",
    "teamspirit": "spirit",
    "lunacyteam": "lunacy",
    "xiesport": "xi",
    "futesports": "fut",
    "donstuesports": "donstu",
    "baksesports": "baks",
    "gremioesports": "gremio",
    "leoteam": "leo",
}


class RatingDateOutOfRange(ValueError):
    def __init__(self, source, observed):
        self.observed = observed
        super().__init__(f"{source} ranking is outdated or future-dated")


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
    origin: str = "provider"


def current_ratings(sources, reference):
    chosen = {}
    for source in sources:
        if not timedelta(0) <= reference - source.as_of <= MAX_RATING_AGE:
            continue
        previous = chosen.get(source.name)
        quality = source.as_of, len(source.teams), source.origin == "provider"
        if previous is None or quality > (previous.as_of, len(previous.teams), previous.origin == "provider"):
            chosen[source.name] = source
    return [chosen[name] for name in SOURCE_ORDER if name in chosen]


def load_ratings(path, origin):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["version"] != 1 or not isinstance(payload["sources"], list):
            return []
        sources = []
        for source in payload["sources"]:
            name = source["name"]
            observed = datetime.fromisoformat(source["as_of"])
            if name not in SOURCE_ORDER or observed.utcoffset() is None:
                continue
            teams = {}
            for key, value in source["teams"].items():
                normalized = team_key(key)
                if not normalized or normalized in teams or type(value) is not int or not 100 <= value <= 4000:
                    break
                teams[normalized] = value
            else:
                if len(teams) >= 2:
                    sources.append(Ratings(name, observed, teams, name == "Fragcharts", origin))
        return sources
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return []


def save_ratings(path, sources):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "sources": [
        {"name": source.name, "as_of": source.as_of.isoformat(), "teams": source.teams}
        for source in sources
    ]}
    temp = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
            temp = Path(output.name)
            json.dump(payload, output, sort_keys=True)
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


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
        raise RatingDateOutOfRange("Fragcharts", observed)
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
        raise RatingDateOutOfRange("CS ELO Analyzer", observed)
    return Ratings("CS ELO Analyzer", observed, {key: value for key, value in ratings.items() if key not in ambiguous})


def predict_ratings(reference):
    ratings, words = parse_table(get_page(PREDICT))
    updated = re.search(r"Last updated\s+(\d{4}-\d\d-\d\d\s+\d\d:\d\d)\s+UTC", words, re.I)
    if len(ratings) < 2 or not updated:
        raise ValueError("CSGOPredict did not return a dated ranking")
    observed = datetime.fromisoformat(" ".join(updated[1].split())).replace(tzinfo=timezone.utc)
    if not timedelta(0) <= reference - observed <= MAX_RATING_AGE:
        raise RatingDateOutOfRange("CSGOPredict", observed)
    return Ratings("CSGOPredict", observed, ratings)
