"""Offline structural, browser-parser and budgeting checks. No betting or network."""
from __future__ import annotations
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path
import pytest

PKG = Path(__file__).resolve().parents[1]


def installer():
    spec = importlib.util.spec_from_file_location('install_moneyline_bankroll', PKG/'install_moneyline_bankroll.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_page(*, variant=False, strategy=True):
    section = '<section><h2>Strategy comparison</h2></section>' if strategy else '<section><h2>Data health</h2></section>'
    render_body = '''\n  renderStrategies();\n  const warnings=[];\n''' if not variant else '''\n  renderStrategies();\n  renderSomeOtherWidget();\n  const warnings = lastData.warnings || [];\n'''
    return '''<!doctype html><html><head><style>body{color:white}</style></head><body><main>
<div id="games" class="cards"></div>
''' + section + '''
<script>
const $=id=>document.getElementById(id);
const el=()=>document.createElement('div');
let lastData={games:[]};
function renderGame(g) {
  const c=el('div');
  title.append(el('h3','',g.away+' @ '+g.home),el('small','muted',date(g.tipoff_utc)));
  m.append(el('b','',win?win.predicted_winner:'Unavailable'));
  c.append(metrics);
}
function render(){''' + render_body + '''}
async function poll(){return 42;}
poll();setInterval(poll,20000);
</script></main></body></html>'''


def patch(page):
    m = installer()
    return m.update_live(page,
        (PKG/'static/moneyline_budget.css').read_text(encoding='utf-8'),
        (PKG/'static/moneyline_budget.js').read_text(encoding='utf-8'))


def test_install_is_additive_idempotent():
    original = sample_page()
    updated, changed = patch(original)
    assert changed
    assert 'id="pg-moneyline-planner"' in updated
    assert updated.count('id="pg-bankroll"') == 1
    assert 'teamMatchup(g.away, g.home)' in updated
    assert 'teamFullName(g.winner_forecast.predicted_winner)' in updated
    assert 'pgDecorateGameCards(lastData.games || []);' in updated
    assert 'renderMoneylinePlanner();' in updated
    assert 'pgBindBudget();' in updated
    assert "title.append(el('h3','',g.away+' @ '+g.home)" in updated  # original renderGame unmodified
    assert 'const warnings=[];' in updated  # existing render body unmodified
    same, again = patch(updated)
    assert not again and same == updated


def test_modified_render_block_no_longer_breaks_install():
    original = sample_page(variant=True)
    assert 'renderStrategies();\n  const warnings=' not in original  # exact prior error
    updated, changed = patch(original)
    assert changed
    assert 'renderSomeOtherWidget();' in updated
    assert 'pgDecorateGameCards(' in updated
    assert updated.index('pgBindBudget();') < updated.index('poll();setInterval(')


def test_fallback_insert_before_health_section():
    updated, changed = patch(sample_page(variant=True, strategy=False))
    assert changed
    assert updated.index('id="pg-moneyline-planner"') < updated.index('<section><h2>Data health</h2>')


def test_bad_html_refuses_to_install():
    with pytest.raises(RuntimeError, match='Incompatible'):
        patch('<html></html>')


def test_does_not_modify_api_or_frozen_files():
    src = (PKG/'install_moneyline_bankroll.py').read_text()
    assert "ROOT/'static/live.html'" in src
    assert "ROOT/'src/api.py'" in src  # existence check only
    assert 'frozen_strategy_v1.json' not in src
    assert 'same_game_paper_track' not in src
    assert 'shutil.copy2(live' in src


def test_nba_logos_fallback():
    js = (PKG/'static/moneyline_budget.js').read_text()
    assert 'https://cdn.nba.com/logos/nba/' in js
    assert 'img.onerror' in js
    assert 'pg-logo-fallback' in js


def test_works_as_python_installer_twice(tmp_path):
    (tmp_path/'src').mkdir()
    (tmp_path/'static').mkdir()
    (tmp_path/'src/api.py').write_text('app = None\n')
    (tmp_path/'static/live.html').write_text(sample_page(variant=True),encoding='utf-8')
    shutil.copy2(PKG/'static/moneyline_budget.css',tmp_path/'static/moneyline_budget.css')
    shutil.copy2(PKG/'static/moneyline_budget.js',tmp_path/'static/moneyline_budget.js')
    shutil.copy2(PKG/'install_moneyline_bankroll.py',tmp_path/'install_moneyline_bankroll.py')
    initial = (tmp_path/'static/live.html').read_bytes()
    for _ in range(2):
        p = subprocess.run(['python', 'install_moneyline_bankroll.py'],cwd=tmp_path,capture_output=True,text=True)
        assert p.returncode == 0,p.stderr+p.stdout
    new = (tmp_path/'static/live.html').read_text()
    assert new.count('id="pg-moneyline-planner"') == 1
    backups = list((tmp_path/'backups').glob('moneyline_bankroll_*/live.html'))
    assert len(backups) == 1
    assert backups[0].read_bytes() == initial


def test_browser_js_syntax_and_budget_math(tmp_path):
    node=shutil.which('node')
    if not node:
        pytest.skip('Node.js not installed')
    js = PKG/'static/moneyline_budget.js'
    subprocess.run([node,'--check',str(js)],check=True)
    run=subprocess.run([node,str(PKG/'tests/validate_moneyline_budget.cjs')],capture_output=True,text=True)
    assert run.returncode == 0,run.stderr+run.stdout
    assert 'passed' in run.stdout
    patched,_=patch(sample_page(variant=True))
    # Confirm our wrapper is legal JavaScript even with changed formatting in render().
    for i,script in enumerate(re.findall(r'<script[^>]*>(.*?)</script>',patched,flags=re.S)):
        path=tmp_path/f'dashboard_{i}.js';path.write_text(script)
        result=subprocess.run([node,'--check',str(path)],capture_output=True,text=True)
        assert result.returncode == 0, result.stderr
