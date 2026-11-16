"""Map a VLM plan (steps) onto the original episode's top-level node order.

Signatures come from the YAML fields (type/name/policies/trajectories/target_object/
source_object/target_holder); nothing per-node is hard-coded. Expects are never rewritten:
an order whose accumulated expect disagrees with planner.simulate is rejected.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import yaml

from .planner import simulate
from .shell.derive import walk

REPO = Path(__file__).resolve().parents[1]
MAPPED = {'policy', 'trajectory', 'lanes', 'terminal'}  # must all appear in the plan
UNSURE = '상태 불확실, 운영자 확인'


def _members(node):
    return [c for lane in node.get('lanes', []) for c in lane.get('nodes', [])] or [node]


def steps_for(order, episode):
    """Inverse of the mapping table: plan steps that the given top-level order executes."""
    sigs, by_id = _signatures(episode), {n['id']: n for n in episode.get('sequence', [])}
    return [list(sigs[c['id']]) for nid in order for c in _members(by_id[nid]) if c['id'] in sigs]


def _signatures(episode):
    """{node_id: (skill, hand, object, target)} for every mappable node, in original order."""
    policies, trajs = episode.get('policies', {}), episode.get('trajectories', {})
    hands = {'right': '-', 'left': '-'}  # held object while walking the original order
    sigs = {}
    for node in walk(episode.get('sequence', [])):
        kind, nid = node.get('type'), node.get('id')
        if kind == 'snapshot':
            sigs[nid] = ('scene_snapshot', '-', '-', '-')
        elif kind == 'terminal':
            sigs[nid] = ('terminal', '-', '-', node.get('result', '-'))
        elif kind == 'policy':
            spec = policies.get(node.get('name'), {})
            side = spec.get('side', '-')
            if spec.get('kind') == 'aglt':
                sigs[nid] = ('grasp', side, node.get('target_object', '-'), '-')
                hands[side] = node.get('target_object', '-')
            elif spec.get('kind') == 'place':
                sigs[nid] = ('place', side, node.get('source_object', '-'), node.get('target_holder', '-'))
                hands[side] = '-'
        elif kind == 'trajectory':
            spec = trajs.get(node.get('name'), {})
            sides = spec.get('sides', [])
            hand = 'both' if sorted(sides) == ['left', 'right'] else (sides[0] if len(sides) == 1 else '-')
            if spec.get('kind') == 'rehome':
                sigs[nid] = ('go_home', hand, '-', '-')
            elif spec.get('kind') == 'carry' and hand in hands:
                sigs[nid] = ('to_tilt_start', hand, hands[hand], '-')
            elif spec.get('kind') == 'tilt':
                sigs[nid] = ('tilt', 'both', hands['right'], hands['left'])
    return sigs


def _hand_world(expect, world):
    for key, side in (('right_hand', 'right'), ('left_hand', 'left')):
        if key in expect:
            world[side] = None if expect[key] == 'EMPTY' else expect[key]


def map_plan(steps, episode, cards=None) -> dict:
    """Return {'ok', 'changed', 'order', 'table', 'reasons'}; order = top-level node ids."""
    top = episode.get('sequence', [])
    original = [n['id'] for n in top]
    out = {'ok': False, 'changed': False, 'order': original, 'table': [], 'reasons': []}
    sim = simulate(steps, cards)
    if not sim['ok']:
        out['reasons'] = ['plan fails simulate: ' + '; '.join(sim['errors'])]
        return out
    sigs = _signatures(episode)
    units = []
    for node in top:
        members = _members(node)
        unsigned = [c['id'] for c in members if node.get('lanes') and c['id'] not in sigs]
        if unsigned:
            out['reasons'].append(f"lanes node {node['id']} has children without a signature: {unsigned}")
            return out
        units.append({'node': node, 'members': members,
                      'sigs': [sigs[c['id']] for c in members if c['id'] in sigs]})
    used, chosen, table = set(), [], []  # chosen: (unit index, last step index)
    i = 0
    while i < len(steps):
        step = tuple(steps[i])
        hit = next((u for u, unit in enumerate(units) if u not in used and step in unit['sigs']), None)
        if hit is None:
            out['reasons'].append(f'step {i} {list(step)}: no unused yaml node with this signature')
            return out
        unit = units[hit]
        k = len(unit['sigs'])
        if Counter(map(tuple, steps[i:i + k])) != Counter(unit['sigs']):
            out['reasons'].append(f"step {i} {list(step)}: lanes node {unit['node']['id']} needs its "
                                  f"{k} children contiguous in the plan")
            return out
        used.add(hit)
        chosen.append((hit, i + k - 1))
        for j in range(i, i + k):
            table.append({'step': list(steps[j]), 'node': unit['node']['id']})
        i += k
    left = [units[u]['node']['id'] for u in range(len(units))
            if u not in used and units[u]['node'].get('type') in MAPPED]
    if left:
        out['reasons'].append(f'yaml nodes missing from plan: {left}')
        return out
    order = [units[u]['node']['id'] for u, _ in chosen]
    # Unmatched passive nodes (extra snapshot/confirm/...) stay right before their original successor.
    for idx in reversed(range(len(top))):
        if idx in used:
            continue
        nid = original[idx]
        nxt = original[idx + 1] if idx + 1 < len(top) else None
        order.insert(order.index(nxt) if nxt in order else len(order), nid)
    out['table'] = table
    out['order'] = order
    out['reasons'] = _check_world(steps, sigs, chosen, units, cards)
    if order != original:
        out['reasons'] += [f"{n['id']}: checkpoint/auto node must keep its position" for i, n in enumerate(top)
                           if (n.get('checkpoint') or n.get('auto')) and order[i] != n['id']]
        confirms = [original[i] for i in range(len(top)) if i not in used and top[i].get('type') == 'confirm']
        if confirms:
            out['reasons'].append(f'confirm nodes not in the plan cannot be reordered around: {confirms}')
        # hand/parallel_policy/... have no signature, so the plan says nothing about where they belong
        opaque = [original[i] for i in range(len(top)) if i not in used and original[i] not in sigs
                  and top[i].get('type') not in MAPPED | {'snapshot', 'confirm'}]
        if opaque:
            out['reasons'].append(f'unmapped nodes without a signature cannot be reordered around: {opaque}')
    out['ok'] = not out['reasons']
    out['changed'] = out['ok'] and order != original
    return out


def _locations(steps, cards):
    return simulate(steps, cards)['world']['locations'] if steps else None  # None = start, all on table


def _check_world(steps, sigs, chosen, units, cards):
    reasons = []
    world = {'right': None, 'left': None}
    first = 0
    for u, last in chosen:
        node, block = units[u]['node'], [tuple(s) for s in steps[first:last + 1]]
        for child in units[u]['members']:
            guard = (child.get('lift_guard') or {}).get('object')
            if guard:
                at = first + block.index(sigs[child['id']])
                locs = _locations(steps[:at], cards)
                if locs is not None and locs.get(guard) != 'table':
                    reasons.append(f"{child['id']}: lift_guard object {guard} not on table at step {at}")
            _hand_world(child.get('expect') or {}, world)
        if node.get('lanes'):
            _hand_world(node.get('expect') or {}, world)
        hands = simulate(steps[:last + 1], cards)['world']['hands']
        if world != hands:
            reasons.append(f"{node['id']}: expect hands {world} != simulate hands {hands}")
        first = last + 1
    return reasons


def _ready_at(interface, order, old_episode_id, timeout):
    """READY on the new runner: status nodes in the requested order and a new episode_id."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = interface.status() or {}
        if (status.get('status') == 'READY' and status.get('node') == order[0]
                and [n.get('id') for n in status.get('nodes') or []] == order
                and status.get('episode_id') not in (None, old_episode_id)):
            return True
        time.sleep(0.1)
    return False


def _not_started(status, ids):
    """Reasons the runner is not at the untouched start of the yaml the mapping used."""
    reasons = []
    if [n.get('id') for n in status.get('nodes') or []] != ids:
        reasons.append('runner nodes != yaml top-level ids')
    if status.get('index') != 0 or (status.get('last') or {}).get('node') or status.get('node') != ids[0]:
        reasons.append('runner already started (index/last/node)')
    return reasons


def apply_plan_order(plan_order, plan, interface, log, mode, ready_timeout=60.0):
    """cli hook. Returns (keep_going, steps the runner will execute or None if unknown).

    keep_going is False whenever apply sent a reorder that was not confirmed applied: a NotSupported/None
    ack, a clean shell rejection before anything ran, or the runner not READY in the new order. We stop
    (and request a runner stop) rather than guess which order is loaded.
    """
    if plan_order == 'off':
        return True, plan['steps']
    event = {'event': 'plan_order', 'plan_order': plan_order, 'mode': mode, 'source': plan.get('source')}
    executed = None
    try:
        status = interface.status() or {}
        path = REPO / 'config' / 'episodes' / f"{status.get('episode')}.yaml"
        raw = path.read_bytes()
        episode = yaml.safe_load(raw)
        ids = [n['id'] for n in episode.get('sequence', [])]
        event.update(episode=str(path), sha256=hashlib.sha256(raw).hexdigest(), episode_id=status.get('episode_id'))
        original = steps_for(ids, episode)
        executed = original if simulate(original)['ok'] else None
        event.update(map_plan(plan['steps'], episode))
        event['reasons'] = event['reasons'] + _not_started(status, ids)
    except Exception as exc:  # mapping must never block the original order
        event.update(ok=False, changed=False, reasons=event.get('reasons', []) + [f'mapping error: {exc!r}'])
    event['applied'] = False
    if plan.get('source') != 'vlm':
        event['reasons'] = event['reasons'] + ['fallback plan is never applied']
    elif plan_order == 'apply' and mode != 'active':
        event['reasons'] = event['reasons'] + ['mode!=active']
    elif plan_order == 'apply' and event.get('changed') and event['reasons']:
        event['reasons'] = event['reasons'] + ['not sent: original order runs']
    elif plan_order == 'apply' and event.get('changed'):
        ack = interface.perform('reorder', active=True, args={'order': event['order'], 'sha256': event['sha256']})
        event['ack'] = ack
        if ack is None or ack.get('error') == 'NotSupported' or not ack.get('ok'):
            if ack is None or ack.get('error') == 'NotSupported':
                event['reasons'] = event['reasons'] + [UNSURE]
            if not (ack or {}).get('stop'):
                event['stop'] = interface.call('stop')
            return False, _final(event, interface, log)
        if not _ready_at(interface, event['order'], event['episode_id'], ready_timeout):
            event['reasons'] = event['reasons'] + ['runner not READY after reorder', UNSURE]
            event['stop'] = interface.call('stop')
            return False, _final(event, interface, log)
        event['applied'] = True
        executed = steps_for(event['order'], episode)
    log(event)
    print(json.dumps({k: event.get(k) for k in ('event', 'plan_order', 'mode', 'ok', 'changed', 'applied',
                                                'sha256', 'reasons')}, ensure_ascii=False), flush=True)
    return True, executed


def _final(event, interface, log):
    """Failure exit: record one more status read so the operator sees where the runner ended."""
    try:
        event['final_status'] = interface.status()
    except Exception as exc:
        event['final_status'] = {'error': repr(exc)}
    log(event)
    print(json.dumps(event, ensure_ascii=False, default=str), flush=True)
    return None
