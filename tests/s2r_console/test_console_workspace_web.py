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
 vm.runInContext(`selectTab('policy')`,ctx);
 assert.equal(el('tab-policy').hidden,false);
 assert.equal(el('tab-setting').hidden,true);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    result = subprocess.run(['node', '-e', script, str(WEB / 'console.js')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
