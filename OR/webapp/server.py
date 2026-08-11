"""전주 산불위험 웹앱 서버 — Python 표준 라이브러리(http.server)만 사용, 무의존성.

브라우저에서 **발화점 CSV·고위험 비율·배분방식**을 입력하면 위험지도가 갱신된다.
물리 위험은 시작 시 1회 계산(또는 캐시 로드)하고, 입력은 지역배율·배분만 다시
적용하므로 초 단위로 반응한다.

실행:
    python webapp/server.py                 # http://localhost:8000
    python webapp/server.py --port 8080
    python webapp/server.py --exposure-v2   # R_base 에 풍하 노출 포함(느린 최초 계산)
    python webapp/server.py --rebuild        # 엔진 캐시 강제 재계산

브라우저에서 http://localhost:<port> 접속.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # engine 모듈
import engine as eng  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("webapp.server")

ENGINE: eng.Engine | None = None  # main 에서 주입.


class Handler(BaseHTTPRequestHandler):
    """GET / → 페이지, POST /api/predict → 위험지도 JSON."""

    def _send(self, code: int, body, ctype: str = "application/json") -> None:
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            html = (HERE / "index.html").read_text(encoding="utf-8")
            self._send(200, html, "text/html")
        elif path == "/health":
            self._send(200, json.dumps({"ok": True, "n_total": ENGINE.n_total}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in ("/api/predict", "/api/spread"):
            self._send(404, json.dumps({"error": "not found"}))
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except Exception:
            req = {}

        if path == "/api/spread":
            self._handle_spread(req)
            return

        prevalence = float(req.get("prevalence", 0.02))
        prevalence = min(max(prevalence, 0.001), 0.2)
        alloc = req.get("alloc", "global")
        if alloc not in ("global", "regime_anchor"):
            alloc = "global"
        fires = req.get("fires") or []
        fires_xy = None
        if fires:
            try:
                arr = np.asarray(fires, dtype=np.float64)
                arr = arr[np.isfinite(arr).all(axis=1)]
                fires_xy = arr if arr.size else None
            except Exception:
                fires_xy = None

        try:
            out = eng.predict(ENGINE, prevalence=prevalence, alloc=alloc,
                              fires_xy=fires_xy)
            self._send(200, json.dumps(out, ensure_ascii=False))
        except Exception as e:  # 방어: 스택은 서버 로그로만.
            logger.exception("predict 실패")
            self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False))

    def _handle_spread(self, req: dict) -> None:
        """POST /api/spread — 발화점에서 풍하 확산 노출확률 지도."""
        try:
            lon = float(req["lon"])
            lat = float(req["lat"])
        except (KeyError, TypeError, ValueError):
            self._send(400, json.dumps({"error": "lon/lat 필요"}, ensure_ascii=False))
            return
        wind_deg = float(req.get("wind_deg", 270.0)) % 360.0
        wind_speed = min(max(float(req.get("wind_speed", 8.0)), 1.0), 30.0)
        try:
            out = ENGINE.spread(lon, lat, wind_deg=wind_deg, wind_speed=wind_speed)
            self._send(200, json.dumps(out, ensure_ascii=False))
        except Exception as e:
            logger.exception("spread 실패")
            self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False))

    def log_message(self, fmt: str, *args) -> None:  # 접근 로그 억제.
        return


def main() -> int:
    global ENGINE
    parser = argparse.ArgumentParser(description="전주 산불위험 웹앱 서버")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--exposure-v2", action="store_true",
                        help="R_base 에 풍하 노출 v2.2 포함(최초 계산 느림).")
    parser.add_argument("--rebuild", action="store_true",
                        help="엔진 캐시 강제 재계산.")
    args = parser.parse_args()

    logger.info("=== 엔진 로드/계산 (최초 1회) ===")
    ENGINE = eng.load_engine(force=args.rebuild, use_exposure_v2=args.exposure_v2)
    logger.info("엔진 준비 완료: 전주 %d본", ENGINE.n_total)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    logger.info("=== 서버 시작: %s (Ctrl-C 로 종료) ===", url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("종료")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
