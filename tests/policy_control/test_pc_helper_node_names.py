"""pd 프로세스의 helper 노드는 런치의 전역 __node remap 에 끌려가면 안 된다 (ros 마커).

launch 가 붙이는 ``--ros-args -r __node:=pd_node`` 는 프로세스 안 **모든** rclpy 노드에
걸린다. ServiceCaller 가 만드는 helper 노드까지 같은 이름이 되면 rcl 이 rosout 퍼블리셔를
공유하고, helper 하나가 destroy 될 때(=engage 의 read_jtc_reference) 그 이름의 rosout
퍼블리셔가 해제되어 **pd 노드 로그가 그 뒤로 발행되지 않는다**. 셀프테스트 직전에 로그가
죽는 자리라 이름 충돌 자체를 금지한다.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.ros

GLOBAL_REMAP = ["--ros-args", "-r", "__node:=pd_node"]


@pytest.fixture
def remapped_ctx():
    """launch 와 같은 전역 노드명 remap 을 건 rclpy 컨텍스트."""
    rclpy = pytest.importorskip("rclpy")
    from rclpy.context import Context

    context = Context()
    rclpy.init(context=context, domain_id=int(os.environ.get("PC_TEST_DOMAIN", "99")),
               args=GLOBAL_REMAP)
    try:
        yield context
    finally:
        rclpy.shutdown(context=context)


def test_service_caller_helper_keeps_its_own_name_under_global_remap(remapped_ctx):
    from rclpy.node import Node

    from policy_control.controller_switch import ServiceCaller

    main = Node("pd_node", context=remapped_ctx)
    caller = ServiceCaller(main, "cm_client", timeout_sec=0.5)
    try:
        assert main.get_name() == "pd_node"
        assert caller.node.get_name() == "pd_node_cm_client", (
            "helper 가 전역 remap 에 끌려가 pd_node 와 이름이 겹친다 — "
            "engage 때 helper 가 destroy 되면 pd 로그가 죽는다")
    finally:
        caller.close()
        main.destroy_node()
