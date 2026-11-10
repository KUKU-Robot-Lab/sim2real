"""Pure episode derivation. All input trees remain untouched."""
from __future__ import annotations

from copy import deepcopy

from ..interface import node_id

VERBS = frozenset({'retry', 'rescan', 'home', 'insert', 'human'})


def walk(nodes):
    for node in nodes:
        yield node
        for lane in node.get('lanes', []):
            yield from walk(lane.get('nodes', []))


def node_side(episode, target):
    """Resolve the actual failed node, including lane children."""
    for node in walk(episode.get('sequence', [])):
        if node.get('id') != target:
            continue
        if node.get('type') == 'policy':
            return episode.get('policies', {}).get(node.get('name'), {}).get('side')
        if node.get('type') == 'trajectory':
            sides = episode.get('trajectories', {}).get(node.get('name'), {}).get('sides', [])
            return sides[0] if len(sides) == 1 else None
        sides = node.get('sides', [])
        return sides[0] if len(sides) == 1 else None
    return None


def derive(original: dict, status: dict, command: dict) -> dict:
    verb, args = command.get('verb'), command.get('args', {})
    if verb not in VERBS or not isinstance(args, dict):
        raise ValueError('invalid shell verb/args')
    result = deepcopy(original)
    sequence = result.get('sequence', [])
    used = {node['id'] for node in walk(sequence)}

    def unique(prefix):
        number = 1
        while f'{prefix}_{number}' in used:
            number += 1
        name = f'{prefix}_{number}'
        used.add(name)
        return name

    world = status.get('world')
    if not isinstance(world, dict):
        raise ValueError('world is required')
    expect = {key: world[key] for key in ('pose', 'right_hand', 'left_hand') if key in world}
    flags = world.get('task_flags', [])
    if isinstance(flags, dict):
        flags = [key for key, value in flags.items() if value is True]
    expect.update({key: True for key in flags})
    held = {world.get('right_hand'), world.get('left_hand')}
    locations = world.get('objects', {})
    objects = [name for name in original.get('objects', {})
               if name not in held and locations.get(name, {}).get('at', 'table')
               not in ('right_hand', 'left_hand')
               and not locations.get(name, {}).get('at', '').startswith('holder:')]
    if not objects:
        raise ValueError('restore snapshot requires at least one observable table object')
    # Explicit objects requires a preceding confirm in the unmodified runner schema.
    prefix = [{'id': unique('shell_confirm'), 'type': 'confirm',
               'text': '복구 전 손 상태와 탁자 물체를 확인하세요'},
              {'id': unique('shell_restore'), 'type': 'snapshot',
               'objects': objects, 'expect': expect}]
    if verb == 'rescan':
        result['sequence'] = prefix + [{'id': unique('shell_done'), 'type': 'terminal', 'result': 'SUCCESS'}]
        return result
    target = args.get('node_id') if verb == 'retry' else node_id(status.get('node'))
    target = target or node_id((status.get('last') or {}).get('node'))
    matches = [i for i, node in enumerate(sequence) if node.get('id') == target]
    if not matches:
        # Retrying a lane child restarts its containing lane group. Partial lane
        # replay cannot preserve the other arm's progress with this YAML schema.
        nested = [i for i, node in enumerate(sequence)
                  if any(child.get('id') == target for child in walk([node]))]
        if nested:
            raise ValueError('lane child retry requires an explicit parent lanes node')
        raise ValueError(f'unknown current/retry node: {target}')
    rest = sequence[matches[0]:]
    if verb == 'home':
        side = args.get('side')
        if side not in ('right', 'left'):
            raise ValueError('home requires side right or left')
        candidates = [node for node in walk(sequence) if node.get('type') == 'trajectory'
                      and result.get('trajectories', {}).get(node.get('name'), {}).get('kind') == 'rehome'
                      and result['trajectories'][node['name']].get('sides') == [side]]
        if not candidates:
            raise ValueError(f'no single-arm rehome trajectory node for {side}')
        home = deepcopy(candidates[0])
        home['id'] = unique(home['id'] + '_shell')
        prefix.append(home)
    elif verb in ('insert', 'human'):
        if verb == 'human':
            text = args.get('text')
            if not isinstance(text, str) or not text.strip():
                raise ValueError('human requires nonempty text')
            spec = {'type': 'confirm', 'text': text}
        else:
            spec = deepcopy(args.get('spec'))
            if not isinstance(spec, dict) or not isinstance(spec.get('type'), str):
                raise ValueError('insert requires a node spec')
        for node in walk([spec]):
            old = node.get('id')
            if not isinstance(old, str) or not old or old in used:
                node['id'] = unique('shell_human' if verb == 'human' else 'shell_' + str(node.get('type', 'insert')))
            else:
                used.add(old)
        prefix.append(spec)
    result['sequence'] = prefix + rest
    return result
