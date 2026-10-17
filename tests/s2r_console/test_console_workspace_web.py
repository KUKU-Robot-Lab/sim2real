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


def test_policy_tab_shows_the_tilt_paths_the_live_confirm_the_checks_and_the_records():
    """10.17 사용자 "policy 부분에 내가 말한 부분들이 시각화 잘되게" — tilt 평가 묶음이 policy 탭에 있고 값이 보인다."""
    html = (WEB / 'index.html').read_text()
    tab = html[html.index('id="tab-policy"'):html.index('id="common-bar"')]
    for pid in ('tilt-panel', 'epf-panel', 'tilt-run-panel', 'tilt-rec-panel'):
        assert f'id="{pid}"' in tab                      # 에피소드 흐름도 policy 탭 안(예전 retired 에서 꺼냄)
    script = r'''
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const els = new Map();
const el = id => { if (!els.has(id)) els.set(id, {id, innerHTML:'', hidden:true, dataset:{}, style:{}, classList:{toggle(){}}, addEventListener(){}}); return els.get(id); };
const ctx = vm.createContext({console, Date, JSON, Set, Map, Number, String, Object, Array, Math,
 document:{getElementById:el, addEventListener(){}, querySelectorAll(){return [];}, body:{classList:{toggle(){}}}},
 localStorage:{getItem(){return 'operator';}},sessionStorage:{getItem(){return 'token';}},
 window:{addEventListener(){}}, setInterval(){}, setTimeout(){}, fetch:async()=>({ok:true,json:async()=>({})}),
 EventSource:function(){this.addEventListener=()=>{}}, requestAnimationFrame(){}});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8').replace('refresh().then(connect);',''),ctx);
const node = (id,type) => ({id,type,title:type,line:id});
const tilt = {errors:[], paths:[
  {key:'A',episode:'tilt_base_a',title:'파지 정책(rg3wo)',when:'먼저',stage:'episode_tilt_base_a',policies:{rh_aglt_r:'rh56f1/aglt/right_rg3wo'},
   nodes:[node('go_home_start','trajectory'),node('grasp_right','policy'),node('tilt','trajectory'),node('done','terminal')],confirms:0,problems:[],ok:true},
  {key:'B',episode:'tilt_base_b',title:'수동 파지(확인 두 번)',when:'A · C 가 안 되면',stage:'episode_tilt_base_b',policies:{},
   nodes:[node('go_home_start','trajectory'),node('confirm_cup_right','confirm'),node('confirm_close_right','confirm'),node('grasp_right','hand'),node('done','terminal')],
   confirms:2,problems:[],ok:true},
  {key:'C',episode:'tilt_base_c',title:'접근 정책',when:'A 4/5 미만',stage:'episode_tilt_base_c',policies:{},nodes:[],confirms:0,problems:['rh_aglt_r: 등록부에 없다'],ok:false}],
 run:{episode:'tilt_base_b',run_dir:'/x/sim2real/logs/episode/r',cards:[
  {node:'to_tilt_start_r',kind:'carry',ok:true,T_end_s:4,rows:[{name:'오른팔 탁자 여유',value:11.96,unit:'cm',limit:1,ok:true},{name:'두 컵 겉면 사이',value:2.1,unit:'cm',limit:3,ok:false}]},
  {node:'grasp_right',kind:'hand',side:'right',ok:false,phase:'failed',why:'not grasped',latched:0}]},
 records:{day:'20261018',summary:{tilt:{A:{n:2,success:1,fill_mean:0.85,spilled_total:2}},grasp:{right:{n:5,success:4}}},
  tilt:[{trial:1,path:'A',in_rcv:10,n_beads:10,in_src:0,spilled:0,success:true,grasp_right:'ok',grasp_left:'ok',video:'a.mov'}],grasp:[],
  criterion:'받는 컵 10/10',grasp_rule:'3 s 유지',command:'eval_record.py tilt'}};
vm.runInContext(`renderTilt(${JSON.stringify({tilt, episode_runner:{episode:'tilt_base_b',phase:'READY',index:1,busy:false,confirm:{node:'confirm_cup_right',text:'병'}}})})`,ctx);
for (const id of ['tilt-panel','tilt-run-panel','tilt-rec-panel']) assert.equal(el(id).hidden,false);
const p = el('tilt-paths').innerHTML;
assert.match(p,/경로 A/); assert.match(p,/✓ 쓸 수 있음/); assert.match(p,/✗ 못 씀/); assert.match(p,/등록부에 없다/);
assert.match(p,/실행 중 · READY/); assert.match(p,/확인 노드 2 개/);
assert.match(p,/tn tn-confirm wait/);                    // 지금 기다리는 확인 노드가 따로 보인다
assert.match(el('tilt-meta').innerHTML,/2\/3 경로/);
const r = el('tilt-run').innerHTML;
assert.match(r,/오른팔 탁자 여유/); assert.match(r,/11\.96/); assert.match(r,/✗ 걸림/); assert.match(r,/not grasped/);
const rec = el('tilt-rec').innerHTML;
assert.match(rec,/1\/2/); assert.match(rec,/4\/5 이상 → 경로 A/); assert.match(rec,/✓ 성공/); assert.match(rec,/받는 컵 10\/10/);
vm.runInContext(`renderTilt({tilt:null})`,ctx);
assert.equal(el('tilt-panel').hidden,true);
'''
    result = subprocess.run(['node', '-e', script, str(WEB / 'console.js')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
