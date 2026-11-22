"""Install historical line calibration into an existing Parlay Guard.

Run from project root after extracting this ZIP. No models, data, or existing
paper-trading records are overwritten. Existing application files are backed up.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

ROOT=Path.cwd()
SOURCE=Path(__file__).resolve().parent/'patch_files'
MARKER='PARLAY_GUARD_LINE_CALIBRATION_V1'

JOINT_HELPER='''# PARLAY_GUARD_LINE_CALIBRATION_V1

def _calibrate_line(line, kind, artifact):
    """Adjust marginal cover P using held-out synthetic-grid validation only.

    The two-sided spread/total case must transform the common greater-than
    event FIRST, then complement. All outputs remain NON-market-validated.
    """
    if line is None:
        return None
    from .calibrate_game_lines import calibrated_probability
    original = float(line['model_probability'])
    if kind == 'spread':
        reverse = line.get('team_side') == 'AWAY'
    else:
        reverse = line['side'].startswith('UNDER ')
    canonical = 1-original if reverse else original
    calibrated, status = calibrated_probability(artifact, kind, canonical)
    probability = 1-calibrated if reverse else calibrated
    new_line = dict(line)
    if kind == 'total' and probability < .5:
        # The conditional most likely total side may flip under calibration.
        before = new_line['side']
        new_line['side'] = ('UNDER ' if before.startswith('OVER ') else 'OVER ') + str(new_line['market_line'])
        new_line['direction'] = -int(new_line['direction'])
        probability = 1-probability
    new_line['model_probability']=round(probability,5)
    new_line['probability_status']=status
    new_line['market_line_validated']=False
    return new_line

'''


def _once(src, old, new, label):
    n=src.count(old)
    if n!=1:
        raise ValueError(f'{label}: expected exactly one anchor, found {n}. No files changed.')
    return src.replace(old,new,1)


def patch_joint(src):
    if MARKER in src:
        return src
    if 'def build_joint_research(snapshot: dict, root: Path)' not in src:
        raise ValueError('Joint Probability Lab is not installed or has an unsupported structure')
    src=_once(src, 'def build_joint_research(snapshot: dict, root: Path) -> dict:',
              JOINT_HELPER+'def build_joint_research(snapshot: dict, root: Path) -> dict:', 'helper anchor')
    src=_once(src,
        "        'singles': [], 'cross_game': [], 'cross_game_line': [], 'same_game': [], 'warnings': [],",
        "        'singles': [], 'cross_game': [], 'cross_game_line': [], 'same_game': [], 'warnings': [],",'payload anchor')
    # Load once; no network and no unsafe pickle. Missing artifact => no claim of calibration.
    before="""    selected = []
    for game in games:"""
    after="""    from .calibrate_game_lines import load_calibrator
    calibrator = load_calibrator(root / 'models/game_line_calibration_v1.json')
    targets = (calibrator or {}).get('targets', {})
    payload['line_calibration'] = {
        'status': ('synthetic_thresholds_evaluated' if calibrator and any(
            v.get('use_for_research') for v in targets.values()) else 'not_validated'),
        'market_line_validated': False,
        'spread_gate': targets.get('home_margin', {}).get('use_for_research', False),
        'total_gate': targets.get('total', {}).get('use_for_research', False),
        'spread_test': targets.get('home_margin', {}).get('test_calibrated'),
        'total_test': targets.get('total', {}).get('test_calibrated'),
        'note': 'Historical evaluation used synthetic forecast-centered thresholds, not observed market lines. Live transfer is unverified.',
    }
    selected = []
    for game in games:"""
    src=_once(src,before,after,'calibration load')
    # Both call-sites are intentionally updated, never a legacy path left silently.
    anchor="line = _line_candidate(other, kind, sigmas[kind])"
    src=_once(src,anchor,"line = _calibrate_line(_line_candidate(other, kind, sigmas[kind]), kind, calibrator)",'cross game line')
    anchor="line = _line_candidate(game, kind, sigmas[kind])"
    src=_once(src,anchor,"line = _calibrate_line(_line_candidate(game, kind, sigmas[kind]), kind, calibrator)",'same game line')
    src=_once(src,
       "'model': 'cross-game independence; game-line cover estimate uncalibrated',",
       "'model': 'cross-game independence; line estimate tested only on synthetic thresholds',\n                    'line_probability_status': line['probability_status'], 'market_line_validated': False,",
       'cross game status')
    src=_once(src,
       "'game_line_probability': line['model_probability'],",
       "'game_line_probability': line['model_probability'],\n                    'line_probability_status': line['probability_status'], 'market_line_validated': False,",
       'same game status')
    src=_once(src,
       "'status': 'ESTIMATED_UNCALIBRATED' if corr else 'INSUFFICIENT_CORRELATION_DATA',",
       "'status': ('ESTIMATED_SYNTHETIC_THRESHOLD_EVALUATED' if corr and line['probability_status'] == 'test_evaluated_synthetic_thresholds'\n                               else 'ESTIMATED_UNCALIBRATED' if corr else 'INSUFFICIENT_CORRELATION_DATA'),",
       'status')
    src=_once(src,
        "    # Cross-game prop+line pairs assume independence across two distinct games.",
        """    # Match the probability base sigma to the validation sigma from the
    # stored calibrator; otherwise the map would act on an unchecked scale.
    for kind, key in (('spread', 'home_margin'), ('total', 'total')):
        target_cal = targets.get(key) or {}
        if target_cal.get('use_for_research'):
            candidate = _finite(target_cal.get('sigma_validation'))
            if candidate is not None and 0 < candidate < 100:
                sigmas[kind] = candidate
    # Cross-game prop+line pairs assume independence across two distinct games.""",
        'calibration sigma match')
    src=src.replace("'2025-26 holdout errors by a normal distribution and are NOT calibrated. '",
                    "'a normal-error model; optional calibration is evaluated ONLY on synthetic thresholds. '",1)
    compile(src,'src/joint_ev.py','exec')
    return src


def patch_html(src):
    if MARKER in src:
        return src
    if 'PARLAY_GUARD_JOINT_RESEARCH_V1' not in src:
        raise ValueError('Joint Probability Lab dashboard was not installed')
    src=_once(src,
        "  const d=lastData.joint_research;",
        """  // PARLAY_GUARD_LINE_CALIBRATION_V1
  const d=lastData.joint_research;""",'html marker')
    src=_once(src,
        "  const items=[",
        """  const cal=d?.line_calibration;
  if(cal){
    const overview=el('div','note small',
      'Line calibration: '+cal.status+' · spread gate '+(cal.spread_gate?'PASS':'NOT VALIDATED')+
      ' · total gate '+(cal.total_gate?'PASS':'NOT VALIDATED')+
      '. Synthetic-threshold evaluation only; actual market-line calibration and fills unverified.');
    root.append(overview);
  }
  const items=[""",'html diagnostics')
    src=_once(src,
        "el('div','gold','Line probability uncalibrated · quote unavailable')",
        "el('div','gold','Line P: '+(r.line_probability_status||'uncalibrated')+' · quote unavailable')",
        'html cross game')
    src=_once(src,
        "el('div','gold','Uncalibrated · executable parlay EV unavailable')",
        "el('div','gold','Line P: '+(r.line_probability_status||'uncalibrated')+' · executable parlay EV unavailable')",
        'html same game')
    src=src.replace('Game-line cover probabilities are uncalibrated approximations from holdout error.',
                    'Line probabilities are evaluated on synthetic historical thresholds, not real market-line quotes.',1)
    return src


def main():
    joint=ROOT/'src/joint_ev.py'; html=ROOT/'static/live.html'
    required=['src/calibrate_game_lines.py','tests/test_line_calibration.py','docs/GAME_LINE_CALIBRATION.md']
    if not joint.is_file() or not html.is_file() or not all((SOURCE/p).is_file() for p in required):
        raise SystemExit('Extract the entire ZIP in the Parlay Guard project root, then retry.')
    old_joint=joint.read_text(encoding='utf-8');old_html=html.read_text(encoding='utf-8')
    if MARKER in old_joint and MARKER in old_html and (ROOT/required[0]).exists():
        print('Line calibration integration already installed; no files changed.')
        return
    if MARKER in old_joint or MARKER in old_html:
        raise SystemExit('Partial installation: refusing to overwrite one-sided changes.')
    new_joint=patch_joint(old_joint);new_html=patch_html(old_html)
    backup=Path.home()/'.parlay_guard_backups'/datetime.now().strftime('%Y%m%d_%H%M%S_line_calibration')
    for path in [joint,html]+[ROOT/r for r in required if (ROOT/r).is_file()]:
        destination=backup/path.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    for name in required:
        to=ROOT/name;to.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(SOURCE/name,to)
    joint.write_text(new_joint,encoding='utf-8')
    html.write_text(new_html,encoding='utf-8')
    print('Installed chronological spread/total calibration research layer.')
    print('Backups:',backup)
    print('Next: python -m pytest tests/ -q')
    print('Then: python -m src.calibrate_game_lines')
    print('Restart Uvicorn to enable the stored calibrator in the Joint Probability Lab.')

if __name__=='__main__':
    main()
