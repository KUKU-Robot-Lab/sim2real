"""Workspace interaction regressions, executing the browser code in Node's VM."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / 'deploy/s2r_console/s2r_console/web'


def test_workspace_tabs_and_stable_video_document():
    html = (WEB / 'index.html').read_text()
    assert 'id="common-controls"' in html
    assert all(f'data-arg="{tab}"' in html for tab in ('setting', 'control', 'policy'))
    assert 'id="fpp-video"' in html
    assert 'id="video-retry"' in html
    assert 'robotmod-panel' not in html


def test_workspace_behavior_in_browser_runtime():
    script = r'''
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const els = new Map();
const el = id => { if (!els.has(id)) els.set(id, {id, innerHTML:'', hidden:true, dataset:{}, style:{}, classList:{toggle(){}}, addEventListener(){}}); return els.get(id); };
const calls=[];
const ctx = vm.createContext({console, Date, JSON, Set, Map, Number, String, Object, Array, Math,
 document:{getElementById:el, addEventListener(){}, querySelectorAll(){return [];}, body:{classList:{toggle(){}}}},
 localStorage:{getItem(){return 'operator';}},sessionStorage:{getItem(){return 'token';}},
 window:{addEventListener(){}}, setInterval(){}, setTimeout(){}, fetch:async(path,opts)=>{calls.push([path,opts && JSON.parse(opts.body)]); return {ok:true,json:async()=>({confirmation_id:'once',title:'home',commands:['hold','home'],warnings:['stop rg3']})};},
 EventSource:function(){this.addEventListener=()=>{}}, requestAnimationFrame(){}});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8').replace('refresh().then(connect);',''),ctx);
(async()=>{
 vm.runInContext(`S={lease:{holder:'operator'}, session:{profile:{domain_class:'fake'}}, workspace:{revision:1,perception:{fpp:true,locate:true},objects:[{id:'src200_pink',enabled:true,input_source:'locate',model:'src200',color:'pink',results:{fpp:{valid:true,frame:'base',position:[1,2,3]},locate:{valid:false,error:'failed'}}}],cards:[],catalog:[]},actions:[{id:'home_right',title:'home',group:'control',enabled:false,reasons:['PD not holding']}]}; renderWorkspace();`,ctx);
 assert.match(el('common-controls').innerHTML,/PD not holding/);
 assert.equal(vm.runInContext(`baseDiagram({cols:[[{id:'arm_drive',title:'팔 구동'},{id:'rh_aglt_node_right',title:'집기 정책'},{id:'cup_holders',title:'홀더'}]],wires:[]}).cols[0].length`,ctx),1);
 assert.match(el('object-cards').innerHTML,/FPP/);
 assert.match(el('object-cards').innerHTML,/locate/);
 assert.match(el('object-cards').innerHTML,/failed/);
 assert.match(el('object-cards').innerHTML,/data-act="object-perception"/);
 await vm.runInContext(`prepareAction('home_right')`,ctx);
 assert.equal(calls[0][0],'/api/actions/prepare');
 assert.match(el('modal-card').innerHTML,/승인 후 실행/);
 assert.doesNotMatch(el('modal-card').innerHTML,/<input/);
 await vm.runInContext(`acts['action-execute']()`,ctx);
 assert.equal(calls[1][0],'/api/actions/execute');
 assert.equal(calls[1][1].confirmation_id,'once');
 await vm.runInContext(`acts['action-execute']()`,ctx);
 assert.equal(calls.filter(c=>c[0]==='/api/actions/execute').length,1);
 vm.runInContext(`S.actions=[{id:'connect_right',title:'c',group:'setting',enabled:true,reasons:[],job:{active:true,stage:'hand_right'},live:{alive:['hand'],dead:[],pd:'IDLE'}},{id:'head_control',title:'h',group:'control',enabled:true,reasons:[]}]; renderWorkspace();`,ctx);
 assert.match(el('setting-actions').innerHTML,/실행 중 · hand_right/);
 assert.match(el('setting-actions').innerHTML,/PD IDLE/);
 assert.match(el('head-settings').innerHTML,/data-act="head-move"/);
 calls.length=0;
 vm.runInContext(`acts['head-preset']('0:-30')`,ctx);
 await vm.runInContext(`acts['head-move']()`,ctx);
 assert.equal(calls[0][0],'/api/actions/prepare');
 assert.deepEqual([calls[0][1].action,calls[0][1].pan,calls[0][1].tilt],['head_control',0,-30]);
 assert.equal(vm.runInContext(`coord4([0.123456,1,-0.5])`,ctx),'[0.1235, 1.0000, -0.5000]');
 vm.runInContext(`selectTab('policy')`,ctx);
 assert.equal(el('tab-policy').hidden,false);
 assert.equal(el('tab-setting').hidden,true);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    result = subprocess.run(['node', '-e', script, str(WEB / 'console.js')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


_VM = r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const els = new Map();
const el = id => { if (!els.has(id)) els.set(id, {id, innerHTML:'', hidden:true, dataset:{}, style:{}, classList:{toggle(){}}, addEventListener(){},
  removeAttribute(){}, getAttribute(){ return ''; }}); return els.get(id); };
const store = {};
const ctx = vm.createContext({console, Date, JSON, Set, Map, Number, String, Object, Array, Math,
 document:{getElementById:el, addEventListener(){}, querySelectorAll(){return [];}, body:{classList:{toggle(){}}}},
 localStorage:{getItem(k){return k in store ? store[k] : 'operator';}, setItem(k, v){store[k] = v;}},sessionStorage:{getItem(){return 'token';}},
 window:{addEventListener(){}}, setInterval(){}, setTimeout(){}, fetch:async()=>({ok:true,json:async()=>({})}),
 EventSource:function(){this.addEventListener=()=>{}}, requestAnimationFrame(){}});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8').replace('refresh().then(connect);',''),ctx);
"""


def _node(script: str):
    result = subprocess.run(['node', '-e', _VM + script, str(WEB / 'console.js')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_node_panel_speaks_plainly_and_timing_hides_for_parallel_pd_profiles():
    """★10.21 사용자 "타이밍 · 노드는 무슨 의미인지 모르겠네" — 노드 칸은 쉬운 말, 사슬이 없는 프로파일은 타이밍 칸을 숨긴다."""
    html = (WEB / 'index.html').read_text()
    assert '<h2>제어 노드 상태</h2>' in html and 'id="metrics-panel"' in html and 'id="col-live"' in html
    _node(r"""
const nodes = [{name:'pd_right', age_s:0.1, stale:false, status:{phase:'TRACKING', ok:true, episode:3, seq:120, proc_ms:0.4}},
               {name:'pd_left', age_s:9, stale:true, status:null}];
const metrics = {episode:1, rows:3, seq_missing:0, budget_ms:8.333, latency_ms:{p50:1, p95:2, max:3}, latency_note:null,
                 proc_ms:{}, series:{latency_ms:[], proc_ms:{}}};
const base = {bridge:{enabled:false, graph:[]}, nodes, metrics};
vm.runInContext(`renderNodes(${JSON.stringify(base)})`, ctx);
const n = el('nodes').innerHTML;
for (const w of ['상태', '명령 번호', '처리 ms', '마지막 수신', '목표 따라감', 'TRACKING', '노드가 떠 있지 않다']) assert.ok(n.includes(w), w);
assert.ok(!/<th>phase<\/th>|>seq<|>proc ms</.test(n));
assert.match(el('bridge-meta').innerHTML, /ROS 연결 꺼짐/);
vm.runInContext(`renderMetrics(${JSON.stringify({...base, profile:{timing:false, status_nodes:['pd_right','pd_left']}})})`, ctx);
assert.equal(el('metrics-panel').hidden, true);
vm.runInContext(`renderMetrics(${JSON.stringify({...base, profile:{timing:true, latency:['obs','pd'], status_nodes:['obs','pd']}})})`, ctx);
assert.equal(el('metrics-panel').hidden, false);
assert.match(el('metrics').innerHTML, /obs → pd/);
""")


def test_policy_tab_keeps_the_tilt_run_checks_and_the_records_folded():
    """10.17 tilt 평가 — ★10.21 경로 카드는 실험 목록으로 옮겼고, 이번 실행 계획 검사 · 회차 기록(접은 칸)은 그대로 보인다."""
    html = (WEB / 'index.html').read_text()
    tab = html[html.index('id="tab-policy"'):html.index('id="common-bar"')]
    for pid in ('epf-panel', 'tilt-run-panel', 'tilt-rec-panel', 'xp-list', 'xp-main', 'pstrip', 'pol-table'):
        assert f'id="{pid}"' in tab
    assert 'id="tilt-panel"' not in html and '<details class="panel fold" id="tilt-rec-panel"' in tab
    _node(r"""
const tilt = {errors:[], paths:[],
 run:{episode:'tilt_base_b',run_dir:'/x/sim2real/logs/episode/r',cards:[
  {node:'to_tilt_start_r',kind:'carry',ok:true,T_end_s:4,rows:[{name:'오른팔 탁자 여유',value:11.96,unit:'cm',limit:1,ok:true},{name:'두 컵 겉면 사이',value:2.1,unit:'cm',limit:3,ok:false}]},
  {node:'grasp_right',kind:'hand',side:'right',ok:false,phase:'failed',why:'not grasped',latched:0}]},
 records:{day:'20261018',summary:{tilt:{A:{n:2,success:1,fill_mean:0.85,spilled_total:2}},grasp:{right:{n:5,success:4}}},
  tilt:[{trial:1,path:'A',in_rcv:10,n_beads:10,in_src:0,spilled:0,success:true,grasp_right:'ok',grasp_left:'ok',video:'a.mov'}],grasp:[],
  criterion:'받는 컵 10/10',grasp_rule:'3 s 유지',command:'eval_record.py tilt'},
 sim:[{kind:'tilt_base',name:'1018_tilt_base_ref',ok:62,n:63,note:'받는 컵 평균 0.998'},{kind:'1c',name:'1018_1c_right',ok:0,n:60,note:'컵 넘어짐 20'}]};
vm.runInContext(`renderTilt(${JSON.stringify({tilt})})`,ctx);
for (const id of ['tilt-run-panel','tilt-rec-panel']) assert.equal(el(id).hidden,false);
const r = el('tilt-run').innerHTML;
assert.match(r,/오른팔 탁자 여유/); assert.match(r,/11\.96/); assert.match(r,/✗ 걸림/); assert.match(r,/not grasped/);
const rec = el('tilt-rec').innerHTML;
assert.match(rec,/1\/2/); assert.match(rec,/4\/5 이상 → 경로 A/); assert.match(rec,/✓ 성공/); assert.match(rec,/받는 컵 10\/10/);
assert.match(rec,/sim 기본 궤적 · 1018_tilt_base_ref/); assert.match(rec,/62\/63/); assert.match(rec,/1-C 경로 C/);
vm.runInContext(`renderTilt({tilt:null})`,ctx);
assert.equal(el('tilt-rec-panel').hidden,true);
""")


def test_policy_tab_experiment_list_shows_the_policy_chain_the_reasons_and_runs_the_chosen_arm():
    """★10.21 사용자 "정책을 찾기 어렵고 실험 진행이 어려움 · 깔끔하고 명확하게": 실험(경로)을 고르면 정책 사슬 ·
    컵 · 왜 못 돌리는지 · 실행 버튼이 한 칸에 모인다. 팔은 목록 한 줄 안에서 고르고, tilt 는 미션 단계 ▶ 로 간다."""
    _node(r"""
const step = (kind, text, sub, extra={}) => ({kind, text, sub, node:'grasp_lift', index:2, sides:['right'], ...extra});
const F = side => ({episode:`grasp_f_${side}`, stage:`episode_grasp_f_${side}`, action:`ctl_grasp_f_${side}`, runner:`episode_runner_${side}`,
  cups:{CUP:'source240b_blue'}, confirms:0, problems:[], summary:'ap05 → rcg04',
  steps:[step('policy','ap05','접근',{id:`rh56f1/aglt/${side}_approach_ap05`}), step('handoff','넘김','사전 파지 자리 2 cm 안'),
         step('policy','rcg04','파지 · 들기',{id:`rh56f1/aglt/${side}_rcg04`})]});
const X = {items:[
  {id:'grasp_b', group:'grasp', key:'1-B', label:'1-B', summary:'1-B 접근 → 확인 → 손 쥐기 → 들기', sides:{right:{...F('right'), episode:'grasp_b_right',
    action:'ctl_grasp_b_right', confirms:1, steps:[step('move','1-B 접근','손 안 컵 자리까지'), step('confirm','확인','병이 손 앞인가')]}}},
  {id:'grasp_f', group:'grasp', key:'F', label:'경로 F', summary:'ap05 → rcg04', sides:{right:F('right'), left:F('left')}},
  {id:'tilt_base_d', group:'tilt', key:'D', label:'tilt D', summary:'1-B 접근 → rcg03 → 이송 → tilt', when:'먼저 — 4/5 이상이면',
   sides:{both:{episode:'tilt_base_d', stage:'episode_tilt_base_d', action:null, runner:'episode_runner', cups:{SRC:'source240b_blue', RCV:'shaker_c_orange'},
     confirms:0, problems:['rh_aglt_l: 등록부에 rh56f1/aglt/left_rcg03 가 없다'], steps:[step('policy','rcg03','파지 · 들기',{id:'rh56f1/aglt/right_rcg03'})]}}}],
 policies:{'rh56f1/aglt/right_rcg04':{id:'rh56f1/aglt/right_rcg04', short:'rcg04', status:'candidate', ep:2800, summary:'파지 · 들기 rcg_04_nr',
   eval:'ap05 출발 98.9 %', checkpoint:'last_ep_2800.pth', contract:'rh_aglt_contract.json', issues:[], path:'deploy/policies/rh56f1/aglt/right_rcg04'},
   'rh56f1/aglt/right_approach_ap05':{id:'rh56f1/aglt/right_approach_ap05', short:'ap05', status:'candidate', ep:2300, summary:'', eval:'', checkpoint:'x', contract:'c', issues:[], path:'p'}}};
const session = {profile:{domain_class:'fake'}, experiments:X, fpp:{tone:'ok', say:'정책 입력으로 써도 된다', stale:false,
  objects:[{name:'source240b_blue', topic:'/objects/source240b_blue/pose', tone:'ok', rows:[{label:'자세 수신', value:'30 Hz', tone:'ok'}]}],
  map:{boxes:[], objects:[{name:'source240b_blue', x:0.25, y:-0.2, tone:'ok', say:'오른팔 범위 안'}]}},
  episode_runners:[{runner:'episode_runner_left', episode:'grasp_f_left', phase:'RUNNING', busy:true, index:2}],
  mission:{rows:[{id:'episode_tilt_base_d', can_run:true, reasons:[], status:'PENDING', touches_real:false}]}};
vm.runInContext(`S={lease:{holder:'operator'}, session:${JSON.stringify(session)}, workspace:{catalog:[
  {id:'rh56f1/aglt/right_rcg04', side:'right', task:'grasping', status:'candidate', checkpoint:'/p/last_ep_2800_rew.pth', issues:[]},
  {id:'rh56f1/place/right_i09', side:'right', task:'placing', status:'candidate', checkpoint:'/p/ep_4000.pth', issues:[]}]},
  actions:[{id:'ctl_grasp_f_right', enabled:false, reasons:['right PD 실시간 상태가 없다'], job:null, live:{alive:[],dead:[]}},
           {id:'ctl_grasp_f_left', enabled:true, reasons:[], job:null, live:{alive:[],dead:[]}},
           {id:'ctl_grasp_b_right', enabled:true, reasons:[], job:null, live:{alive:[],dead:[]}}]};
  render = () => { renderExperiments(S.session); renderPolicyTable(S.session); renderFpp(S.session); }; render();`, ctx);   // 가짜 DOM — 스크롤 고정 없이
// 고른 것이 없으면 지금 도는 실험(경로 F 왼팔 실행기가 돈다)을 연다
let list = el('xp-list').innerHTML, main = el('xp-main').innerHTML;
assert.match(list, /data-act="xp-pick" data-arg="grasp_f" aria-pressed="true"/);
assert.match(list, /파지 · 들기/); assert.match(list, /tilt<small>두 팔/);
assert.match(list, /xp-dot live[^>]*>L</);                          // 경로 F 왼팔 = 실행 중
assert.match(main, /경로 F/); assert.match(main, /rcg04/); assert.match(main, /ep 2800 · 후보/); assert.match(main, /사전 파지 자리 2 cm 안/);
assert.match(main, /source240b_blue/); assert.match(main, /\(0\.250, -0\.200\)/); assert.match(main, /확인 노드 없음/);
assert.match(main, /right PD 실시간 상태가 없다/); assert.match(main, /data-act="action-prepare" data-arg="ctl_grasp_f_right" disabled/);
// 정책 칸을 누르면 체크포인트 · 계약 · sim 근거가 열린다
vm.runInContext(`acts['xp-pol']('rh56f1/aglt/right_rcg04')`, ctx);
main = el('xp-main').innerHTML;
assert.match(main, /class="xp-pol"/); assert.match(main, /sim 근거/); assert.match(main, /98\.9/); assert.match(main, /last_ep_2800\.pth/);
// 왼팔로 바꾸면 그 팔 작업 · 실행기 — 실행 중이라 실행 칸이 그렇게 말한다
vm.runInContext(`acts['xp-side']('left')`, ctx);
main = el('xp-main').innerHTML;
assert.match(main, /data-arg="ctl_grasp_f_left"/); assert.match(main, /⟳ 실행 중/);
assert.equal(vm.runInContext('epRunner', ctx), 'episode_runner_left');      // 지금 에피소드 칸도 그 실행기
assert.match(store['s2r.xp'], /"side":"left"/);
// 1-B — 확인 노드가 있으면 그렇게, 실행할 수 있으면 버튼이 켜진다
vm.runInContext(`acts['xp-pick']('grasp_b')`, ctx);
main = el('xp-main').innerHTML;
assert.match(main, /확인 노드 1 개/); assert.match(main, /✓ 실행할 수 있다/);
assert.match(main, /data-act="action-prepare" data-arg="ctl_grasp_b_right" >/);
// tilt — 미션 단계 ▶, 파일이 막으면(등록부 없음) 단계가 준비돼도 막는다
vm.runInContext(`acts['xp-pick']('tilt_base_d')`, ctx);
main = el('xp-main').innerHTML;
assert.match(main, /data-act="run" data-arg="episode_tilt_base_d" disabled/); assert.match(main, /등록부에 rh56f1\/aglt\/left_rcg03 가 없다/);
assert.match(main, /먼저 — 4\/5 이상이면/); assert.match(main, /<small>SRC<\/small>/);
// 정책 목록 — 등록부 전부, 쓰는 실험은 그 실험으로 바로 간다
const pol = el('pol-table').innerHTML;
assert.match(pol, /rcg04/); assert.match(pol, /i09/); assert.match(pol, /data-act="xp-pick" data-arg="grasp_f">경로 F/);
assert.match(el('pol-meta').innerHTML, /2 개 · 실험에 쓰는 것 1/);
// FP++ 한 줄 — 판정 · 물체 자리 · 한마디, 영상은 누를 때만
const strip = el('pstrip').innerHTML;
assert.match(strip, /정책 입력으로 써도 된다/); assert.match(strip, /source240b_blue/); assert.match(strip, /오른팔 범위 안/);
assert.equal(el('fpp-panel').hidden, true);
vm.runInContext(`acts['xp-video']()`, ctx);
assert.equal(el('fpp-panel').hidden, false);
""")


def test_experiments_view_reads_the_chain_from_the_episode_files():
    """★10.21 실험 목록의 사슬은 에피소드 파일 · 등록부에서 — 손으로 적은 제목(옛 'tilt D = ap05 → rcg03 넘김')과 어긋나지 않는다."""
    import sys
    import yaml
    sys.path.insert(0, str(ROOT / 'deploy/policy_control'))
    sys.path.insert(0, str(ROOT / 'deploy/s2r_console'))
    from s2r_console import experiments as XV

    ids = [st['id'] for st in yaml.safe_load((ROOT / 'config/mission_rh56f1_fake.yaml').read_text())['stages']]
    v = XV.view(ROOT, ids)
    assert not v.get('error')
    items = {i['id']: i for i in v['items']}
    assert [i['key'] for i in v['items'] if i['group'] == 'grasp'] == ['1-B', 'C', 'D', 'E', 'F']
    assert [i['key'] for i in v['items'] if i['group'] == 'tilt'] == ['D', 'C', 'B']
    f = items['grasp_f']['sides']['right']
    assert [(s['kind'], s['text']) for s in f['steps']] == [('policy', 'ap05'), ('handoff', '넘김'), ('policy', 'rcg04')]
    assert f['action'] == 'ctl_grasp_f_right' and f['runner'] == 'episode_runner_right' and f['cups'] == {'CUP': 'source240b_blue'}
    assert '2 cm' in f['steps'][1]['sub'] and '0.03 m/s' in f['steps'][1]['sub']
    assert items['grasp_c']['summary'] == 'ap05 → 스크립트 쥐기'
    assert items['grasp_b']['sides']['left']['confirms'] == 1
    d = items['tilt_base_d']['sides']['both']
    assert d['action'] is None and d['stage'] == 'episode_tilt_base_d' and d['runner'] == 'episode_runner'
    assert items['tilt_base_d']['summary'].startswith('1-B 접근 → rcg03')
    p = v['policies']['rh56f1/aglt/right_rcg04']
    assert p['short'] == 'rcg04' and p['ep'] == 2800 and p['path'] == 'deploy/policies/rh56f1/aglt/right_rcg04'
    assert XV.short_name('rh56f1/aglt/left_approach_ap05') == 'ap05'


def test_tilt_view_reads_the_sim_baselines_and_the_run_checks(tmp_path):
    import json, sys
    sys.path.insert(0, str(ROOT / 'deploy/policy_control'))
    sys.path.insert(0, str(ROOT / 'deploy/s2r_console'))
    from s2r_console import tilt_view as TV
    (tmp_path / 'logs/sim/a').mkdir(parents=True)
    (tmp_path / 'logs/sim/a/judge.json').write_text(json.dumps({'summary': {'n': 4, 'n_valid': 3, 'success_end': 2, 'in_rcv_mean': 0.9,
                                                                           'spill_total': 1, 'success_any': 2}}))
    (tmp_path / 'logs/sim/b').mkdir(parents=True)
    (tmp_path / 'logs/sim/b/result.json').write_text(json.dumps({'cfg': {'stop_palm_m': 0.06}, 'cond': 'train', 'judge': 'contact', 'rows': [
        {'script_ok': True, 'true_lift': True, 'd_stop_true_m': 0.05, 'why': ''},
        {'script_ok': True, 'true_lift': False, 'd_stop_true_m': 0.07, 'why': "['tipped']"}]}))
    for name, judge in (('c', 'none'), ('d', None)):
        (tmp_path / 'logs/sim' / name).mkdir(parents=True)
        (tmp_path / 'logs/sim' / name / 'result.json').write_text(json.dumps({'cfg': {'stop_palm_m': 0.06}, 'cond': 'real', 'rows': [],
                                                                           **({'judge': judge} if judge else {})}))
    sim = {x['name']: x for x in TV.sim_view(tmp_path)}
    assert (sim['a']['ok'], sim['a']['n']) == (2, 3) and (sim['b']['ok'], sim['b']['n']) == (1, 2) and '넘어짐 1' in sim['b']['note'] and '판정 접촉' in sim['b']['note']
    assert '강제 파지' in sim['c']['note'] and '래치 판정' in sim['d']['note']                # 10.18 기본 = 강제 파지 · 옛 결과 = 래치
    run = tmp_path / 'run'
    (run / 'carry_to_tilt_start_r').mkdir(parents=True)
    (run / 'carry_to_tilt_start_r/check.json').write_text(json.dumps({'ok': False, 'T_end_s': 2.0, 'sides': ['right'], 'checks': {
        'right': {'table_clear_m': 0.005, 'vmax_rad_s': 0.4, 'arrived': True, 'plan_ok': True, 'end_q_err_rad': 0.0, 'end_cup_err_mm': 0.1}},
        'arms_clear': {'min_arm_point_m': 0.3, 'min_cup_gap_m': 0.2, 'ok': True}}))
    v = TV.run_view({'episode': 'tilt_base_a', 'run_dir': str(run), 'nodes': [{'id': 'to_tilt_start_r'}]})
    (card,) = v['cards']
    assert card['kind'] == 'carry' and not card['ok']
    assert [r['ok'] for r in card['rows'] if '탁자' in r['name']] == [False]          # 탁자 여유 0.5 cm < 1 cm
    assert TV.run_view({'episode': 'grasp_rg3_right', 'run_dir': str(run)}) is None
