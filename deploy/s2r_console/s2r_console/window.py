"""콘솔을 브라우저 대신 **자기 창**으로 연다 — GTK 3 + WebKit2GTK (시스템 패키지, 새 의존성 없음).

화면은 브라우저 모드와 같은 web/ 을 그대로 쓴다. HTTP 서버는 127.0.0.1 에 그대로 떠 있어서
원격 열람(ssh -L)도 계속 된다 — 창은 그 주소를 여는 전용 뷰어일 뿐이다.

창을 닫으면 콘솔이 내려간다(Ctrl+C 와 같은 정리 경로). 실행 중인 단계·자식이 있으면 먼저 묻는다 —
브라우저 탭은 닫아도 서버가 남았지만, 창은 닫는 순간 콘솔이 끝나기 때문이다.
"""
from __future__ import annotations

import signal
from typing import Any, Callable

WEBKIT_VERSIONS = ("4.1", "4.0")
APP_ID = "s2r-console"


def load() -> tuple[Any, Any, Any]:
    """(Gtk, GLib, WebKit2). 없으면 ImportError — 설치 명령을 담는다."""
    try:
        import gi

        gi.require_version("Gtk", "3.0")
        for ver in WEBKIT_VERSIONS:
            try:
                gi.require_version("WebKit2", ver)
                break
            except ValueError:
                continue
        else:
            raise ImportError("WebKit2 GI 바인딩이 없다")
        from gi.repository import GLib, Gtk, WebKit2  # type: ignore[attr-defined]
    except (ImportError, ValueError) as exc:
        raise ImportError(f"{exc} — sudo apt install gir1.2-webkit2-4.1 (또는 브라우저 모드로 띄울 것)") from exc
    return Gtk, GLib, WebKit2


def run(url: str, *, title: str, close_warning: Callable[[], str]) -> None:
    """창을 띄우고 닫힐 때까지 막는다. `close_warning()` 이 빈 문자열이 아니면 닫기 전에 확인을 받는다."""
    Gtk, GLib, WebKit2 = load()
    GLib.set_prgname(APP_ID)                          # 작업 표시줄·.desktop 의 StartupWMClass 와 짝
    win = Gtk.Window(title=title)
    win.set_default_size(1600, 1000)
    view = WebKit2.WebView()
    view.load_uri(url)
    win.add(view)

    def on_delete(*_):
        warning = close_warning()
        if not warning:
            return False
        dlg = Gtk.MessageDialog(transient_for=win, modal=True, message_type=Gtk.MessageType.WARNING,
                                buttons=Gtk.ButtonsType.OK_CANCEL, text="콘솔을 닫을까?")
        dlg.format_secondary_text(warning)
        answer = dlg.run()
        dlg.destroy()
        return answer != Gtk.ResponseType.OK          # True 면 닫기를 막는다

    win.connect("delete-event", on_delete)
    win.connect("destroy", lambda *_: Gtk.main_quit())
    for sig in (signal.SIGINT, signal.SIGTERM):       # 터미널 Ctrl+C · kill 도 창을 닫고 같은 정리로 간다
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, lambda *_: (Gtk.main_quit(), False)[1])
    win.show_all()
    Gtk.main()


def notice(title: str, text: str) -> None:
    """창이 닫힌 뒤 남길 말(실기 드라이버를 남겼다 등) — 터미널이 없을 수 있으므로 대화상자로도 띄운다."""
    Gtk, _, _ = load()
    dlg = Gtk.MessageDialog(message_type=Gtk.MessageType.WARNING, buttons=Gtk.ButtonsType.OK, text=title)
    dlg.format_secondary_text(text)
    dlg.run()
    dlg.destroy()
