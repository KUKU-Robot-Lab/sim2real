from __future__ import annotations

import pytest

from s2r_console.lease import Lease, LeaseError


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_second_operator_is_refused_until_expiry():
    clock = Clock()
    lease = Lease(clock=clock, ttl_s=30)
    lease.acquire("a")
    with pytest.raises(LeaseError, match="a 가 조작 중"):
        lease.acquire("b")
    clock.t = 30.0
    assert lease.acquire("b").holder == "b"


def test_same_operator_keeps_the_same_token():
    lease = Lease(clock=Clock())
    assert lease.acquire("a").token == lease.acquire("a").token


def test_force_takes_over_and_the_old_token_dies():
    lease = Lease(clock=Clock())
    old = lease.acquire("a").token
    new = lease.acquire("b", force=True).token
    assert new != old
    with pytest.raises(LeaseError):
        lease.check(old)
    assert lease.check(new).holder == "b"


def test_renew_extends_and_a_wrong_token_does_not():
    clock = Clock()
    lease = Lease(clock=clock, ttl_s=30)
    tok = lease.acquire("a").token
    clock.t = 20.0
    lease.renew(tok)
    clock.t = 45.0
    assert lease.view()["holder"] == "a"
    with pytest.raises(LeaseError):
        lease.renew("nope")


@pytest.mark.parametrize("token", [None, "", "wrong"])
def test_check_refuses_bad_tokens(token):
    lease = Lease(clock=Clock())
    lease.acquire("a")
    with pytest.raises(LeaseError):
        lease.check(token)


def test_no_lease_at_all_is_refused_and_blank_names_too():
    lease = Lease(clock=Clock())
    with pytest.raises(LeaseError, match="lease 가 없다"):
        lease.check("x")
    with pytest.raises(LeaseError, match="비어"):
        lease.acquire("  ")


def test_release_frees_it():
    lease = Lease(clock=Clock())
    tok = lease.acquire("a").token
    lease.release(tok)
    assert lease.view() == {"holder": None, "expires_in_s": None}
