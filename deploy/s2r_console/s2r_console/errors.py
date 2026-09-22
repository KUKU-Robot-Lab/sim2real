"""패키지 안에서 같이 쓰는 예외 — 모듈끼리 서로를 import 하지 않게 따로 둔다."""
from __future__ import annotations


class ProfileError(ValueError):
    """프로파일 yaml 이 틀렸다. 로드 시점에 난다 — 틀린 프로파일로 run 을 열지 않는다."""
