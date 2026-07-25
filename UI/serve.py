"""UI mockup preview server (no-cache), port 8001.

Usage:  python serve.py  ->  http://localhost:8001
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    print("BodyBuddy UI mockup server: http://localhost:8001 (no-cache)")
    ThreadingHTTPServer(("", 8001), NoCacheHandler).serve_forever()
