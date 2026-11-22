"""Idempotent live dashboard enhancement: logos and moneyline budget preview.

Only edits static/live.html. Preserves all existing API, frozen strategy files,
pre-tipoff picks, same-game comparisons and model files.
"""
from __future__ import annotations
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
MARKER = 'PARLAY_GUARD_BANKROLL_ALLOCATOR_V1'
SECTION = '''<section id="pg-moneyline-planner" class="pg-planner" aria-label="Today's moneyline bankroll allocation">
<div class="head"><div><div class="eyebrow">Moneyline singles · Research preview</div><h2>Today’s bankroll allocation</h2><p class="pg-sub">Enter your starting bankroll. Only positive model-EV picks receive a suggested paper stake; the rest stay at $0. No bets are placed.</p></div><span class="pill" id="pg-day">Today</span></div>
<div class="pg-budget-controls">
<label>Starting bankroll ($)<input id="pg-bankroll" type="number" value="100" min="1" max="10000000" step="10" inputmode="decimal"></label>
<label>Already committed today ($)<input id="pg-committed" type="number" value="0" min="0" step="0.01" inputmode="decimal"></label>
<label>Max risk today<select id="pg-daily-pct"><option value="0.02">2% of bankroll</option><option value="0.05" selected>5% of bankroll</option><option value="0.10">10% of bankroll</option></select></label>
<label>Max risk per game<select id="pg-game-pct"><option value="0.01">1% of bankroll</option><option value="0.02" selected>2% of bankroll</option></select></label>
<label>Minimum estimated EV<select id="pg-min-ev"><option value="0">Above 0%</option><option value="0.05" selected>At least 5%</option><option value="0.10">At least 10%</option></select></label>
</div>
<div class="pg-summary"><div class="pg-sum"><small>Daily risk ceiling</small><strong id="pg-cap">—</strong></div><div class="pg-sum"><small>Already committed</small><strong id="pg-committed-total">—</strong></div><div class="pg-sum"><small>New paper allocation</small><strong id="pg-total">—</strong></div><div class="pg-sum"><small>Remaining risk allowance</small><strong id="pg-left">—</strong></div><div class="pg-sum"><small>Funded / today’s games</small><strong id="pg-count">—</strong></div></div>
<div id="pg-moneyline-results" aria-live="polite"></div>
<p class="pg-risk-note">Sizing: quarter-Kelly from Elo probability and observed best ask, limited by the per-game cap, daily cap and displayed top-of-book shares. Quotes can move and fills are not assured. EV estimates are unvalidated and can be wrong. Enter stakes already committed today to avoid double-counting the daily cap after earlier games disappear from the live page. Unused bankroll is deliberately not forced into negative-edge bets. No orders are submitted; changing these controls does not modify frozen experiments.</p>
</section>'''


DECORATOR = """
// PARLAY_GUARD_BANKROLL_ALLOCATOR_V2: add logos and allocations without rewriting existing render functions.
function pgDecorateGameCards(games) {
  const cards = document.querySelectorAll('#games > .card');
  for (let i = 0; i < cards.length && i < games.length; i++) {
    const card = cards[i], g = games[i];
    const heading = card.querySelector('.head h3');
    if (heading) heading.replaceWith(teamMatchup(g.away, g.home));
    const winner = card.querySelector('.winner b');
    if (winner && g.winner_forecast) winner.textContent = teamFullName(g.winner_forecast.predicted_winner);
    let stake = card.querySelector('[data-pg-moneyline-slug]');
    if (!stake) {
      stake = el('div', 'pg-recommendation');
      stake.dataset.pgMoneylineSlug = g.slug;
      const metrics = card.querySelector('.metrics');
      if (metrics) metrics.insertAdjacentElement('afterend', stake);
      else card.append(stake);
    }
  }
}
const pgOriginalRender = render;
render = function (...args) {
  const returned = pgOriginalRender.apply(this, args);
  pgDecorateGameCards(lastData.games || []);
  renderMoneylinePlanner();
  return returned;
};
pgBindBudget();
"""


def update_live(source: str, css: str, js: str) -> tuple[str, bool]:
    """Add a separate panel and hook, without relying on formatting inside render().

    Local dashboard variants may add widgets or rearrange the render function;
    that must not prevent the bankroll preview from installing.
    """
    if 'id="pg-moneyline-planner"' in source:
        if ('pgDecorateGameCards(' in source and 'renderMoneylinePlanner();' in source
                and 'pgBindBudget();' in source):
            return source, False
        if 'data-pg-moneyline-slug' in source and 'renderMoneylinePlanner();' in source:
            # Original V1 installer already completed successfully.
            return source, False
        raise RuntimeError('An incomplete bankroll panel is present; refusing to duplicate it. Restore its backed-up static/live.html first.')

    # Only rely on top-level dashboard structure and API state bindings.
    required = ['</style>', 'id="games"', 'function render(',
                'lastData', 'function renderGame(', '</script>']
    missing = [x for x in required if x not in source]
    if missing:
        raise RuntimeError('Incompatible static/live.html; missing required dashboard elements: '
                           + ', '.join(missing))

    # Preserve the original page intact; CSS, panel, and one JS hook are additive.
    updated = source.replace('</style>', css + '\n</style>', 1)
    strategies = '<section><h2>Strategy comparison</h2>'
    if strategies in updated:
        updated = updated.replace(strategies, SECTION + '\n' + strategies, 1)
    elif '<section><h2>Data health</h2>' in updated:
        anchor = '<section><h2>Data health</h2>'
        updated = updated.replace(anchor, SECTION + '\n' + anchor, 1)
    elif '</main>' in updated:
        updated = updated.replace('</main>', SECTION + '\n</main>', 1)
    else:
        raise RuntimeError('Incompatible static/live.html; cannot locate dashboard panel insertion point')

    script_start = updated.index('function render(')
    script_end = updated.find('</script>', script_start)
    if script_end == -1:
        raise RuntimeError('Incompatible static/live.html; dashboard render script has no closing tag')
    # Prefer insertion before the initial poll, so the first result renders with
    # names/logos/stakes. If the polling syntax changed, append to its script.
    import re
    poll = re.search(r'\bpoll\s*\(\s*\)\s*;', updated[script_start:script_end])
    insert_at = script_start + poll.start() if poll else script_end
    updated = updated[:insert_at] + '\n' + js + '\n' + DECORATOR + '\n' + updated[insert_at:]
    return updated, True


def main() -> None:
    live = ROOT/'static/live.html'
    css = ROOT/'static/moneyline_budget.css'
    js = ROOT/'static/moneyline_budget.js'
    if not (ROOT/'src/api.py').is_file() or not live.is_file():
        raise SystemExit('Run from Parlay Guard root (must contain src/api.py and static/live.html).')
    for path in [css,js]:
        if not path.is_file():
            raise SystemExit('Extract ZIP contents before installation: missing '+str(path))
    original = live.read_text(encoding='utf-8')
    updated,changed = update_live(original,css.read_text(encoding='utf-8'),js.read_text(encoding='utf-8'))
    if changed:
        backup=ROOT/'backups'/('moneyline_bankroll_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        backup.mkdir(parents=True,exist_ok=True)
        shutil.copy2(live,backup/'live.html')
        temp=live.with_suffix('.pg-tmp')
        temp.write_text(updated,encoding='utf-8')
        temp.replace(live)
        print('Live dashboard updated. Original saved to:',backup/'live.html')
    else:
        print('Moneyline bankroll calculator already installed; no files changed.')
    print('Preserved frozen moneyline + same-game tracker configurations and recorded picks.')
    print('Preview (no betting): http://127.0.0.1:8000/')

if __name__=='__main__':
    main()
