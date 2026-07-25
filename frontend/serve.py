"""개발용 정적 서버 (캐시 비활성화).

python -m http.server는 브라우저가 JS 모듈을 캐시해서 수정이 반영 안 되는
문제가 있다. 이 서버는 Cache-Control: no-store를 붙여 항상 최신 파일을 준다.

사용법:  python serve.py  →  http://localhost:8000
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    print("BodyBuddy dev server: http://localhost:8000 (no-cache)")
    ThreadingHTTPServer(("", 8000), NoCacheHandler).serve_forever()
