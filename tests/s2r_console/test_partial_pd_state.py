"""A connected arm must not make a never-connected opposite arm unknown."""
from types import SimpleNamespace as NS

from s2r_console.console_state import Observed
from s2r_console.diagram_spec import Box


def test_partial_pd_liveness_is_scoped_to_requested_side(console):
    boxes = tuple(Box(id=f'pd_{side}', title=side, col=0, status=f'pd_{side}',
                      units=(f'pd_arm_{side}#1',), ros=(f'/pd_{side}',)) for side in ('right', 'left'))
    session = NS(diagram=NS(boxes=boxes), profile=NS(status_nodes=('pd_right', 'pd_left'), is_real=True))
    observed = Observed(status={'pd_right': {'phase': 'TRACKING'}}, age_s={'pd_right': 0},
                        rosgraph={'nodes': ['/pd_right']}, rosgraph_age_s=0)
    procs = {'pd_arm_right#1': {'alive': True}}
    assert console._pd_phase(session, observed, procs, side='left') is None
    assert console._pd_phase(session, observed, procs, side='right') == 'TRACKING'
    procs['pd_arm_left#1'] = {'alive': True}
    assert console._pd_phase(session, observed, procs, side='left') == 'UNKNOWN'
