"""The video route relays the existing MJPEG feed without accepting browser URLs."""
import http.client
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from s2r_console import server


def test_video_forwards_frames_and_reports_connection(console, monkeypatch):
    class Camera(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
            self.end_headers()
            self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8example\xff\xd9\r\n')

    camera = ThreadingHTTPServer(('127.0.0.1', 0), Camera)
    threading.Thread(target=camera.serve_forever, daemon=True).start()
    monkeypatch.setenv('S2R_VIDEO_URL', f'http://127.0.0.1:{camera.server_port}/stream')
    console.open('t_fake', operator='op')
    httpd = server.serve(console, bind='127.0.0.1', port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection('127.0.0.1', httpd.server_port, timeout=3)
        conn.request('GET', '/api/video', headers={'Host': 'localhost'})
        response = conn.getresponse()
        assert response.status == 200
        assert 'multipart/x-mixed-replace' in response.getheader('Content-Type')
        assert b'\xff\xd8example\xff\xd9' in response.read()
        assert console.video_status()['last_frame_at'] is not None
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        camera.shutdown()
        camera.server_close()


def test_video_rejects_foreign_host_before_opening_upstream(console):
    httpd = server.serve(console, bind='127.0.0.1', port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection('127.0.0.1', httpd.server_port, timeout=3)
        conn.request('GET', '/api/video', headers={'Host': 'evil.example'})
        assert conn.getresponse().status == 403
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
