"""단일 쓰기 lease — 두 브라우저가 동시에 engage 를 누르는 일을 막는다.

안전장치가 **아니다**(그건 승인 원장과 pd 의 watchdog 이다). 경합 방지다. 읽기는 lease 없이 언제나 된다.
순수하다 — 시계를 주입받는다.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Callable

TTL_S = 30.0


class LeaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class Held:
    holder: str
    token: str
    expires: float


class Lease:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic, ttl_s: float = TTL_S) -> None:
        self._clock, self._ttl = clock, ttl_s
        self._held: Held | None = None

    def _live(self) -> Held | None:
        if self._held is not None and self._clock() >= self._held.expires:
            self._held = None
        return self._held

    def acquire(self, holder: str, *, force: bool = False) -> Held:
        holder = holder.strip()
        if not holder:
            raise LeaseError("운영자 이름이 비어 있다")
        live = self._live()
        if live is not None and live.holder != holder and not force:
            raise LeaseError(f"{live.holder} 가 조작 중이다 ({live.expires - self._clock():.0f} s 남음)")
        token = live.token if (live is not None and live.holder == holder) else secrets.token_hex(8)
        self._held = Held(holder, token, self._clock() + self._ttl)
        return self._held

    def renew(self, token: str) -> Held:
        live = self.check(token)
        self._held = Held(live.holder, live.token, self._clock() + self._ttl)
        return self._held

    def check(self, token: str | None) -> Held:
        live = self._live()
        if live is None:
            raise LeaseError("조작 lease 가 없다 — 먼저 잡아야 한다")
        if not token or not secrets.compare_digest(token, live.token):
            raise LeaseError(f"{live.holder} 가 조작 중이다")
        return live

    def release(self, token: str | None) -> None:
        self.check(token)
        self._held = None

    def view(self) -> dict:
        live = self._live()
        if live is None:
            return {"holder": None, "expires_in_s": None}
        return {"holder": live.holder, "expires_in_s": round(live.expires - self._clock(), 1)}
