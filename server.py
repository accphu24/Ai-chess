"""
Chess AI Arena - bản web (FastAPI). Chạy một ván tại một thời điểm.

Chạy local:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...        # chỉ cần key của dịch vụ bạn định dùng
    uvicorn server:app --port 8080      # rồi mở http://localhost:8080

Khi đưa lên mạng, đặt ACCESS_TOKEN để chỉ mình bạn dùng được (tránh người lạ
đốt quota API của bạn).
"""

import datetime
import hmac
import os
import threading
import time

import chess
import chess.pgn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from players import make_player

ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Chess AI Arena")


def auth(x_token: str = Header(default="")):
    if ACCESS_TOKEN and not hmac.compare_digest(x_token, ACCESS_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")


class Game:
    def __init__(self, white, black, white_spec, black_spec, delay, max_plies):
        self.players = {chess.WHITE: white, chess.BLACK: black}
        self.specs = {chess.WHITE: white_spec, chess.BLACK: black_spec}
        self.delay = delay
        self.max_plies = max_plies
        self.board = chess.Board()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.paused = False
        self.status = "running"  # running | finished | stopped | error
        self.message = ""
        self.result = "*"
        self.thinking = None  # chess.WHITE / chess.BLACK / None
        self.last = None
        self.log = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    # ---- trạng thái ----
    def active(self):
        return self.status == "running"

    def snapshot(self):
        with self.lock:
            return {
                "status": self.status,
                "paused": self.paused,
                "message": self.message,
                "result": self.result,
                "fen": self.board.fen(),
                "last": self.last,
                "white": self.players[chess.WHITE].name,
                "black": self.players[chess.BLACK].name,
                "thinking": None if self.thinking is None else ("white" if self.thinking else "black"),
                "log": list(self.log),
            }

    def pgn(self):
        with self.lock:
            game = chess.pgn.Game.from_board(self.board)
            game.headers["Event"] = "Chess AI Arena"
            game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
            game.headers["White"] = self.players[chess.WHITE].name
            game.headers["Black"] = self.players[chess.BLACK].name
            game.headers["Result"] = self.result
            return str(game)

    # ---- vòng lặp ván cờ ----
    def _sleep(self, seconds):
        end = time.time() + seconds
        while time.time() < end and not self.stop_event.is_set():
            time.sleep(0.1)

    def _fail(self, msg):
        with self.lock:
            self.status = "error"
            self.message = msg
            self.thinking = None

    def run(self):
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    over = self.board.is_game_over(claim_draw=True) or len(self.board.move_stack) >= self.max_plies
                    paused = self.paused
                if over:
                    with self.lock:
                        outcome = self.board.outcome(claim_draw=True)
                        if outcome:
                            self.result = outcome.result()
                            self.message = outcome.termination.name
                        else:
                            self.result = "*"
                            self.message = "ĐẠT GIỚI HẠN SỐ NƯỚC"
                        self.status = "finished"
                        self.thinking = None
                    return
                if paused:
                    time.sleep(0.2)
                    continue

                with self.lock:
                    turn = self.board.turn
                    board_copy = self.board.copy()
                    self.thinking = turn
                player = self.players[turn]
                try:
                    move, note = player.choose_move(board_copy)
                except Exception as e:  # lỗi mạng, hết quota, key sai...
                    self._fail(f"{player.name}: {e}")
                    return
                if self.stop_event.is_set():
                    return
                with self.lock:
                    if move not in self.board.legal_moves:
                        self.status = "error"
                        self.message = f"{player.name} trả về nước không hợp lệ"
                        self.thinking = None
                        return
                    number = self.board.fullmove_number
                    san = self.board.san(move)
                    self.log.append({"n": number, "white": turn == chess.WHITE, "san": san,
                                     "who": player.name, "note": note or ""})
                    self.board.push(move)
                    self.last = move.uci()
                    self.thinking = None
                self._sleep(self.delay)
        finally:
            for p in self.players.values():
                p.close()


current = None
GLOCK = threading.Lock()


class StartReq(BaseModel):
    white: str = "random"
    black: str = "random"
    delay: float = 1.0
    max_plies: int = 200


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.api_route("/healthz", methods=["GET", "HEAD"], response_class=PlainTextResponse)
def healthz():
    """Không cần mã truy cập: dành cho health check của Render và UptimeRobot."""
    return "ok"


_models_cache = {"t": 0.0, "data": None}
_EXCLUDE = ("whisper", "tts", "guard", "embed", "imagen", "veo", "image", "aqa", "audio",
            "orpheus", "playai", "moderation", "transcribe", "live", "robotics")
PROVIDER_LABELS = {"gemini": "Gemini", "groq": "Groq", "openrouter": "OpenRouter (miễn phí)", "ollama": "Ollama (local)"}


def _list_models(provider):
    from openai import OpenAI
    from players import PRESETS

    base_url, key_env = PRESETS[provider]
    api_key = os.environ.get(key_env, "") if key_env else "ollama"
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=8 if key_env else 2)
    ids = [m.id.replace("models/", "", 1) for m in client.models.list().data]
    if provider == "openrouter":
        ids = [i for i in ids if i.endswith(":free")]
    elif provider == "gemini":
        ids = [i for i in ids if i.startswith(("gemini", "gemma"))]
    ids = [i for i in ids if not any(x in i.lower() for x in _EXCLUDE)]
    return sorted(set(ids))


@app.get("/api/models", dependencies=[Depends(auth)])
def models(refresh: bool = False):
    """Danh sách lựa chọn cho menu: chỉ hiện dịch vụ đã có API key, lấy model-id thật từ dịch vụ."""
    now = time.time()
    if not refresh and _models_cache["data"] and now - _models_cache["t"] < 600:
        return _models_cache["data"]
    groups = [{"label": "Cơ bản", "items": [
        {"spec": "random", "label": "Random (ngẫu nhiên)"},
        {"spec": "stockfish", "label": "Stockfish (engine)"},
    ]}]
    notes = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        groups.append({"label": "Claude", "items": [
            {"spec": "claude:claude-sonnet-5-5", "label": "Claude Sonnet 5.5"},
            {"spec": "claude:claude-haiku-5-5", "label": "Claude Haiku 5.5"},
            {"spec": "claude:claude-opus-5-5", "label": "Claude Opus 5.5"},
        ]})
    else:
        notes.append("Chưa đặt ANTHROPIC_API_KEY")
    from players import PRESETS
    for provider in ("gemini", "groq", "openrouter", "ollama"):
        key_env = PRESETS[provider][1]
        if key_env and not os.environ.get(key_env):
            notes.append(f"Chưa đặt {key_env}")
            continue
        if provider == "ollama" and os.environ.get("RENDER"):
            continue  # Ollama không chạy được trên Render
        try:
            ids = _list_models(provider)
        except Exception as e:
            if provider != "ollama":
                notes.append(f"{PROVIDER_LABELS[provider]}: không lấy được danh sách ({type(e).__name__})")
            continue
        if ids:
            groups.append({"label": PROVIDER_LABELS[provider],
                           "items": [{"spec": f"{provider}:{i}", "label": i} for i in ids]})
    data = {"groups": groups, "notes": notes}
    _models_cache.update(t=now, data=data)
    return data


@app.get("/api/state", dependencies=[Depends(auth)])
def state():
    if current is None:
        return {"status": "idle", "paused": False, "message": "", "result": "*",
                "fen": chess.STARTING_FEN, "last": None, "white": "", "black": "",
                "thinking": None, "log": []}
    return current.snapshot()


@app.post("/api/start", dependencies=[Depends(auth)])
def start(req: StartReq):
    global current
    with GLOCK:
        if current and current.active():
            raise HTTPException(409, "Đang có ván chạy. Hãy dừng ván hiện tại trước.")
        try:
            white = make_player(req.white.strip())
        except (Exception, SystemExit) as e:
            raise HTTPException(400, f"Không tạo được người chơi Trắng: {e}")
        try:
            black = make_player(req.black.strip())
        except (Exception, SystemExit) as e:
            white.close()
            raise HTTPException(400, f"Không tạo được người chơi Đen: {e}")
        delay = max(0.0, min(30.0, req.delay))
        max_plies = max(10, min(400, req.max_plies))
        current = Game(white, black, req.white, req.black, delay, max_plies)
        current.thread.start()
    return {"ok": True}


@app.post("/api/pause", dependencies=[Depends(auth)])
def pause():
    if current is None or not current.active():
        raise HTTPException(409, "Không có ván nào đang chạy.")
    with current.lock:
        current.paused = not current.paused
        return {"paused": current.paused}


@app.post("/api/stop", dependencies=[Depends(auth)])
def stop():
    if current is None or not current.active():
        raise HTTPException(409, "Không có ván nào đang chạy.")
    current.stop_event.set()
    with current.lock:
        current.status = "stopped"
        current.thinking = None
    return {"ok": True}


@app.get("/api/pgn", dependencies=[Depends(auth)], response_class=PlainTextResponse)
def pgn():
    if current is None:
        raise HTTPException(404, "Chưa có ván nào.")
    return current.pgn()
