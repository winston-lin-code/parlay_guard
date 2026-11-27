"""Record public game state and Polymarket US market depth for paper execution."""

import argparse
import asyncio
import base64
import getpass
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen
from zoneinfo import ZoneInfo

GATEWAY = 'https://gateway.polymarket.us'
MARKETS_PATH = '/v1/ws/markets'
MARKETS_WS = 'wss://api.polymarket.us' + MARKETS_PATH
KEY_ID_VAR = 'POLYMARKET_US_KEY_ID'
SECRET_VAR = 'POLYMARKET_US_SECRET_KEY'
ET = ZoneInfo('America/New_York')


def parse_time(value):
    if value.endswith('Z'):
        value = value[:-1] + '+00:00'
    return datetime.fromisoformat(value)


def fetch(path):
    with urlopen(GATEWAY + path, timeout=8) as response:
        return json.load(response)


def event_detail(slug):
    return fetch('/v1/events/slug/' + quote(slug, safe=''))['event']


def upcoming_events(league, now):
    league = quote(league, safe='')
    listed = fetch(f'/v2/leagues/{league}/events?limit=50')['events']
    return [
        event for event in listed
        if event.get('startTime') and not event.get('ended') and parse_time(event['startTime']) > now
    ]


def resolve_events(args):
    if args.event:
        return [event_detail(args.event)]
    now = datetime.now(timezone.utc)
    if args.next:
        upcoming = upcoming_events(args.next, now)
        if not upcoming:
            raise ValueError(f'No upcoming {args.next} game is listed')
        return [min(upcoming, key=lambda event: parse_time(event['startTime']))]

    day = args.date or now.astimezone(ET).date()
    games = {}
    for league in dict.fromkeys(args.leagues):
        upcoming = [
            event for event in upcoming_events(league, now)
            if parse_time(event['startTime']).astimezone(ET).date() == day
        ]
        if not upcoming:
            print(f'No upcoming {league.upper()} games on {day:%b %d} ET', flush=True)
        for event in upcoming:
            games[event['slug']] = event
    return sorted(games.values(), key=lambda event: parse_time(event['startTime']))

def read_env_file(path):
    values = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        key, separator, value = line.strip().partition('=')
        key = key.strip()
        if not separator or key not in (KEY_ID_VAR, SECRET_VAR):
            continue
        if key in values:
            raise ValueError(f'{path} declares {key} more than once')
        value = value.strip()
        if value.startswith(('"', "'")):
            delimiter = value[0]
            value, closing, tail = value[1:].partition(delimiter)
            if not closing or (tail.strip() and not tail.lstrip().startswith('#')):
                raise ValueError(f'Invalid quoted value for {key} in {path}')
        else:
            value = value.partition('#')[0].strip()
        values[key] = value
    missing = [key for key in (KEY_ID_VAR, SECRET_VAR) if not values.get(key)]
    if missing:
        raise ValueError(f'{path} is missing ' + ', '.join(missing))
    return values[KEY_ID_VAR], values[SECRET_VAR]


def get_credentials(prompt_key, env_file=None):
    if env_file is None:
        key_id = os.getenv(KEY_ID_VAR)
        secret = os.getenv(SECRET_VAR)
    else:
        key_id, secret = read_env_file(env_file)
    if prompt_key and not (key_id and secret):
        if not sys.stdin.isatty():
            raise ValueError('--prompt-key requires an interactive terminal')
        key_id = key_id or input('Polymarket US Key ID: ').strip()
        secret = secret or getpass.getpass('Polymarket US secret key: ').strip()
    if not (key_id and secret):
        raise ValueError(f'Set {KEY_ID_VAR} and {SECRET_VAR}, use --env-file or --prompt-key, or choose --mode poll')
    return signing_credentials(key_id, secret)


def signing_credentials(key_id, secret):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    seed = base64.b64decode(secret, validate=True)
    if len(seed) not in (32, 64):
        raise ValueError('Polymarket US secret key must decode to 32 or 64 bytes')
    return key_id, Ed25519PrivateKey.from_private_bytes(seed[:32])


def auth_headers(credentials):
    key_id, private_key = credentials
    timestamp = str(time.time_ns() // 1_000_000)
    signature = private_key.sign(f'{timestamp}GET{MARKETS_PATH}'.encode())
    return {
        'X-PM-Access-Key': key_id,
        'X-PM-Timestamp': timestamp,
        'X-PM-Signature': base64.b64encode(signature).decode(),
    }


def game_state(event):
    state = event.get('eventState') or (event.get('metadata') or {}).get('gameState') or {}
    return {
        'period': state.get('period') or event.get('period'),
        'clock': state.get('elapsed') or event.get('elapsed'),
        'score': state.get('score') or event.get('score'),
        'live': state.get('live', event.get('live')),
        'ended': state.get('ended', event.get('ended')),
        'source_state_timestamp': state.get('updatedAt'),
    }


async def poll_game(slug, interval, write, ended):
    while True:
        try:
            starting_ns = time.time_ns()
            event = await asyncio.to_thread(event_detail, slug)
            state = game_state(event)
            write('game_state', {'http_start_ns': starting_ns, **state})
            if state['ended']:
                ended.set()
                return
        except (URLError, TimeoutError, ValueError) as error:
            write('game_state_error', {'error_type': type(error).__name__})
        await asyncio.sleep(interval)


async def poll_book(markets, interval, write):
    async def one_book(slug):
        try:
            starting_ns = time.time_ns()
            path = f'/v1/markets/{quote(slug, safe="")}/book'
            payload = await asyncio.to_thread(fetch, path)
            write('book_snapshot', {'http_start_ns': starting_ns, 'book': payload['marketData']})
        except (URLError, TimeoutError, ValueError) as error:
            write('book_error', {'market': slug, 'error_type': type(error).__name__})
    while True:
        await asyncio.gather(*(one_book(slug) for slug in markets))
        await asyncio.sleep(interval)


async def stream_books(markets, credentials, write):
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed, InvalidStatus

    subscribe = {'subscribe': {
        'requestId': 'pregame-paper-capture',
        'subscriptionType': 'SUBSCRIPTION_TYPE_MARKET_DATA',
        'marketSlugs': markets,
    }}
    while True:
        try:
            async with connect(MARKETS_WS, additional_headers=auth_headers(credentials), open_timeout=8) as ws:
                await ws.send(json.dumps(subscribe))
                write('subscription_sent', {'markets': markets})
                confirmed = False
                while True:
                    frame = await asyncio.wait_for(ws.recv(), timeout=45 if confirmed else 10)
                    message = json.loads(frame)
                    if message.get('error'):
                        raise ValueError('The exchange rejected the market subscription')
                    data = message.get('marketData')
                    if data and data.get('marketSlug') in markets:
                        if not confirmed:
                            write('subscription_confirmed', {'market': data['marketSlug']})
                            print(f"Receiving WebSocket book updates for {data['marketSlug']}", flush=True)
                            confirmed = True
                        write('book_update', {'book': data})
        except InvalidStatus as error:
            status = error.response.status_code
            if status in (401, 403):
                raise ValueError(f'Polymarket US rejected WebSocket authentication (HTTP {status})') from None
            write('book_connection_error', {'http_status': status})
        except (ConnectionClosed, OSError, TimeoutError) as error:
            write('book_connection_error', {'error_type': type(error).__name__})
        await asyncio.sleep(2)


async def capture(args, event, credentials):
    slug = event['slug']
    start = parse_time(event['startTime'])
    start_et = start.astimezone(ET)
    print(f"{event['title']} starts {start_et:%b %d at %I:%M %p %Z}", flush=True)
    if args.lead_minutes is not None:
        connect_at = start.timestamp() - 60*args.lead_minutes
        if connect_at > time.time():
            print(f"Connecting at {datetime.fromtimestamp(connect_at, ET):%b %d at %I:%M %p %Z}", flush=True)
        while connect_at > time.time():
            await asyncio.sleep(min(30, connect_at-time.time()))

    # League listings can omit markets. Load the full event shortly before capture.
    event = await asyncio.to_thread(event_detail, slug)
    featured = [
        m for m in event.get('markets', [])
        if m.get('active') and not m.get('closed') and (
            str(m.get('sportsMarketType', '')).endswith('_team_full_game_winner')
            or m.get('sportsMarketType') == 'soccer_team_full_time_winner'
        )
    ]
    if not featured:
        raise ValueError(f'No active full-game moneyline is listed for {slug}')
    market_slugs = [m['slug'] for m in featured]
    print('Markets: ' + ', '.join(market_slugs), flush=True)
    output = args.output or Path(f"{slug}-{time.time_ns()}-{args.mode}.jsonl")
    print(f'{args.mode} capture: {output.resolve()}', flush=True)
    ended = asyncio.Event()
    with output.open('x', encoding='utf-8', buffering=1) as log:
        def write(kind, data):
            log.write(json.dumps({'kind': kind, 'received_at_ns': time.time_ns(), 'event': slug, **data},
                                 separators=(',', ':')) + '\n')

        write('session', {'mode': args.mode, 'start_time': event['startTime'],
                          'markets': [{'slug': m['slug'], 'tick_size': m.get('orderPriceMinTickSize')}
                                      for m in featured]})
        book = stream_books(market_slugs, credentials, write) if args.mode == 'ws' else poll_book(
            market_slugs, args.poll_seconds, write
        )
        tasks = {asyncio.create_task(coro) for coro in (book, poll_game(slug, args.poll_seconds, write, ended), ended.wait())}
        try:
            done, _ = await asyncio.wait(tasks, timeout=args.minutes*60, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        write('session_end', {'game_ended': ended.is_set()})


async def capture_games(args, events, credentials):
    async def one_game(event):
        try:
            await capture(args, event, credentials)
            return True
        except (ValueError, URLError, OSError, ImportError) as error:
            print(f"{event['slug']}: {error}", file=sys.stderr, flush=True)
            return False
    return all(await asyncio.gather(*(one_game(event) for event in events)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--event', help='Polymarket US event slug')
    target.add_argument('--next', help='pick the next scheduled game in this league')
    target.add_argument('--leagues', nargs='+', help='record the upcoming games in these leagues on one ET date')
    parser.add_argument('--date', type=date.fromisoformat, help='ET date YYYY-MM-DD for --leagues, defaults to today')
    parser.add_argument('--list', action='store_true', help='print the selected schedule without starting a capture')
    parser.add_argument('--mode', choices=('ws', 'poll'), default='ws')
    credential_source = parser.add_mutually_exclusive_group()
    credential_source.add_argument('--prompt-key', action='store_true', help='prompt for a missing Key ID or secret')
    credential_source.add_argument('--env-file', type=Path, help='read both Polymarket US credentials from this .env file')
    parser.add_argument('--lead-minutes', type=float, help='wait to connect until this many minutes before start')
    parser.add_argument('--minutes', type=float, default=120, help='capture duration from connection')
    parser.add_argument('--poll-seconds', type=float, default=1)
    parser.add_argument('--output', type=Path, help='new JSONL file for the book and game-state data')
    args = parser.parse_args()
    if args.minutes <= 0 or args.poll_seconds < .25 or (args.lead_minutes is not None and args.lead_minutes < 0):
        parser.error('Use a positive duration, a polling interval of at least 0.25 seconds, and a nonnegative lead time')
    if args.date and not args.leagues:
        parser.error('--date requires --leagues')
    if args.leagues and args.lead_minutes is None:
        args.lead_minutes = 15
    try:
        events = resolve_events(args)
        if args.list:
            for event in events:
                start_et = parse_time(event['startTime']).astimezone(ET)
                capture_et = datetime.fromtimestamp(start_et.timestamp() - 60*(args.lead_minutes or 0), ET)
                print(f"{event['title']} | {event['slug']} | starts {start_et:%b %d at %I:%M %p %Z}"
                      f" | capture {capture_et:%I:%M %p %Z}")
            return
        if not events:
            return
        if args.output and len(events)>1:
            parser.error('--output requires a single game')
        credentials = get_credentials(args.prompt_key, args.env_file) if args.mode == 'ws' else None
        if not asyncio.run(capture_games(args, events, credentials)):
            parser.exit(1)
    except (ValueError, URLError, OSError, ImportError) as error:
        parser.exit(1, f'{error}\n')
    except KeyboardInterrupt:
        parser.exit(130)


if __name__ == '__main__':
    main()
