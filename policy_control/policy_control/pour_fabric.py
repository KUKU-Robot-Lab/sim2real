"""Bimanual pour fabric step: palm target (6) + synergy hand target -> fabric joint state, per role.

Mirror of hdgp ``pour_fabric/side_rig.py`` (setup_fabric / sync_fabric_hand / step_fabric / pin_fabric)
and ``pour_fabric_env.py::_build_fabric_world``. Both roles share one world (a single table box that is
the union of both palm boxes). Everything comes from ``PourContract``; the fabrics_sim backend types and
helpers are reused from ``fabric_core``.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from .fabric_core import (ORIENTATION, PALM_DIM, PCA_DIM, FabricBackend, FabricError, JointTarget,
                          _fabric_class, _verify_backend, name_permutation)
from .pour_contract import PourContract


def pour_world_dict(contract: PourContract) -> dict | None:
    """Same box the training env builds; None when the run had no table obstacle."""
    f = contract.fabric
    if not f.table_obstacle:
        return None
    lo = [min(s.box_lo[i] for s in contract.sides) for i in range(2)]
    hi = [max(s.box_hi[i] for s in contract.sides) for i in range(2)]
    m, th = float(f.table_margin_xy), float(f.table_thickness)
    sx, sy = (hi[0] - lo[0]) + 2 * m, (hi[1] - lo[1]) + 2 * m
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    cz = float(f.table_z) - 0.5 * th
    return {"table": {"env_index": "all", "type": "box", "scaling": f"{sx} {sy} {th}",
                      "transform": f"{cx} {cy} {cz} 0. 0. 0. 1."}}


def make_pour_world(contract: PourContract, device: str):
    """(object_ids, object_indicator) of the shared world (CUDA only)."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise FabricError(f"fabrics_sim needs an available CUDA device, got {device!r}")
    from fabrics_sim.utils.utils import initialize_warp
    from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

    initialize_warp(str(device)[-1])
    world = WorldMeshesModel(batch_size=1, device=device,
                             max_objects_per_env=int(contract.fabric.max_objects),
                             world_dict=pour_world_dict(contract))
    return world, world.get_object_ids()


def make_pour_fabric(contract: PourContract, role: str, device: str, world_ids) -> FabricBackend:
    """Real fabrics_sim backend for one role, built with the env's constructor arguments."""
    from fabrics_sim.integrator.integrators import DisplacementIntegrator

    s, f = contract.side(role), contract.fabric
    kw = {"fabric_params_filename": s.fabric_params} if s.fabric_params else {}
    fab = _fabric_class(s.fabric_class)(
        batch_size=1, device=device, timestep=float(contract.fabric_dt), graph_capturable=False,
        use_hand_fabric=False, tip_per_finger=False, hand_mode="pca",
        use_hand_repulsion=bool(f.use_hand_repulsion),
        use_body_repulsion_pairs=bool(f.use_body_repulsion_pairs),
        robot_dir_name=s.fabric_robot_dir, robot_name=s.fabric_robot_dir, **kw)
    return FabricBackend(fabric=fab, integrator=DisplacementIntegrator(fab),
                         object_ids=world_ids[0], object_indicator=world_ids[1], device=str(device))


class PourFabricCore:
    """One role's fabric trajectory generator (state: q, qd, qdd in ``fabric_joint_order``)."""

    def __init__(self, contract: PourContract, role: str, device: str, home_q, *,
                 backend: FabricBackend) -> None:
        self.contract = contract
        self.role = role
        self.side_cfg = s = contract.side(role)
        self.backend = backend
        self.device = backend.device
        self._order = tuple(s.fabric_joint_order)
        self._n = len(self._order)
        self._n_arm = len(s.arm_joints)
        if self._order[:self._n_arm] != tuple(s.arm_joints):
            raise FabricError(f"{role}: fabric_joint_order does not start with the arm joints")
        _verify_backend(backend, self._order)
        self._perm = torch.as_tensor(name_permutation(s.hand_joints, self._order[self._n_arm:]),
                                     dtype=torch.long, device=self.device)
        self._pca = torch.zeros(1, PCA_DIM, device=self.device)
        self._damping = float(contract.fabric.damping) * torch.ones(1, 1, device=self.device)
        self.reset(home_q)

    @property
    def joint_names(self) -> tuple:
        return self._order

    @property
    def q(self) -> np.ndarray:
        return self._q[0].detach().cpu().numpy().astype(np.float64)

    def _row(self, values, n: int, what: str) -> torch.Tensor:
        if values is None:
            raise FabricError(f"{self.role}: {what} is required")
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size != n or not np.all(np.isfinite(arr)):
            raise FabricError(f"{self.role}: {what} must be {n} finite values, got {arr.size}")
        return torch.as_tensor(arr, device=self.device).unsqueeze(0)

    def reset(self, q_home) -> None:
        """Seed the state and the cspace rest posture (env: ``default_config.copy_(fabric_q)``)."""
        self._q = self._row(q_home, self._n, "home_q").contiguous()
        self._qd = torch.zeros_like(self._q)
        self._qdd = torch.zeros_like(self._q)
        self.backend.fabric.default_config.copy_(self._q)

    def set_state(self, q, qd) -> None:
        """Overwrite (q, qd) without moving the cspace rest posture (replay / re-seed from measurement)."""
        self._q = self._row(q, self._n, "q").contiguous()
        self._qd = self._row(qd, self._n, "qd").contiguous()
        self._qdd = torch.zeros_like(self._q)

    def step(self, palm6, hand_target, hold: bool = False) -> JointTarget:
        """hand sync -> set_features -> integrate x decimation -> (hold) pin back, as the env does."""
        palm = self._row(palm6, PALM_DIM, "palm target")
        hand = self._row(hand_target, len(self.side_cfg.hand_joints), "hand target")
        q = self._q.clone()
        q[:, self._n_arm:] = hand[:, self._perm]
        q_pin, qd, qdd = q, self._qd, self._qdd
        b, dt = self.backend, float(self.contract.fabric_dt)
        with torch.inference_mode(False), torch.no_grad():
            b.fabric.set_features(self._pca, palm, ORIENTATION, q.detach(), qd.detach(),
                                  b.object_ids, b.object_indicator, self._damping)
            subs = []
            for _ in range(int(self.contract.fabric_decimation)):
                q, qd, qdd = b.integrator.step(q.detach(), qd.detach(), qdd.detach(), dt)
                subs.append(q[0].detach().cpu().numpy().astype(np.float64))
        if hold:
            q, qd, qdd = q_pin, torch.zeros_like(qd), torch.zeros_like(qdd)
            subs = [q[0].detach().cpu().numpy().astype(np.float64)] * len(subs)
        self._q, self._qd, self._qdd = q.detach().clone(), qd.detach().clone(), qdd.detach().clone()
        full = self.q
        qd_arm = self._qd[0, :self._n_arm].cpu().numpy().astype(np.float64)
        return JointTarget(q_arm=full[:self._n_arm].copy(),
                           qd_arm=qd_arm * float(self.contract.fabric.vel_ff_scale),
                           q_full=full, substeps=np.stack(subs))


class PourFabricPair:
    """Both roles on one shared world. ``home`` / ``backends`` are keyed by role."""

    def __init__(self, contract: PourContract, device: str, home: Mapping, *,
                 backends: Mapping | None = None) -> None:
        self.contract = contract
        if backends is None:
            self._world, ids = make_pour_world(contract, device)     # keep the world alive
            backends = {r: make_pour_fabric(contract, r, device, ids) for r in contract.roles}
        missing = [r for r in contract.roles if r not in backends or r not in home]
        if missing:
            raise FabricError(f"missing backend/home for roles {missing}")
        self.cores = {r: PourFabricCore(contract, r, device, home[r], backend=backends[r])
                      for r in contract.roles}

    def reset(self, home: Mapping) -> None:
        for r, core in self.cores.items():
            core.reset(home[r])

    def step(self, palm: Mapping, hand: Mapping, hold: bool = False) -> dict:
        return {r: core.step(palm[r], hand[r], hold) for r, core in self.cores.items()}
