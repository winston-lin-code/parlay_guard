"""Installer uses stable anchors and declines unexpected project layouts."""
import importlib.util
from pathlib import Path
import pytest


def load_installer():
    # This is a preinstallation test, so it must find the root installer from
    # either the project root (after extraction) or the tests' parent.
    path=Path(__file__).resolve().parents[1]/'install_market_line_validation.py'
    spec=importlib.util.spec_from_file_location('marketline_installer',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_api_patch_adds_safe_readonly_endpoint_once():
    m=load_installer()
    base="app = object()\n@app.get('/api/health')\ndef health(): return {}\n"
    changed=m.patch_api(base)
    assert '/api/market-line-validation' in changed
    assert m.patch_api(changed)==changed
    compile(changed,'src/api.py','exec')


def test_dashboard_patch_is_idempotent_and_escapes_dynamic_text():
    m=load_installer()
    html=('''<section><h2>Data health</h2></section>\n<script>
const el=(tag,cls,txt)=>document.createElement(tag);
async function poll(){let lastData = null;lastData=await r.json();render();}
</script>''')
    changed=m.patch_html(html)
    assert 'market-audit-summary' in changed
    assert 'renderMarketLineAudit()' in changed
    assert 'lastData=await r.json();render();renderMarketLineAudit()' in changed
    assert m.patch_html(changed)==changed


def test_unexpected_source_refused():
    m=load_installer()
    with pytest.raises(ValueError,match='expected one anchor'):
        m.patch_api('no expected FastAPI route')
    with pytest.raises(ValueError,match='expected one anchor'):
        m.patch_html('different HTML')
