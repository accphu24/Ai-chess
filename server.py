"""
Chess AI Arena - bản web (FastAPI). Chạy một ván tại một thời điểm.

Chạy local:
    pip install -r requirements.txt
    export GEMINI_API_KEY=...           # chỉ cần key của dịch vụ bạn định dùng
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

from players import PRESETS, make_player

ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Chess AI Arena")


def auth(x_token: str = Header(default="")):
    if ACCESS_TOKEN and not hmac.compare_digest(x_token, ACCESS_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")


class Game:
    """phase: idle (giữa 2 nước) | thinking (AI đang nghĩ) | ready (AI đã nghĩ xong, chờ bấm Tiếp tục)
    | error (AI gặp lỗi, chờ bấm Tiếp tục để thử lại)."""

    def __init__(self, white, black, delay, max_plies, manual):
        self.players = {chess.WHITE: white, chess.BLACK: black}
        self.delay = delay
        self.max_plies = max_plies
        self.manual = manual
        self.board = chess.Board()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.go = threading.Event()  # được set khi người dùng bấm "Tiếp tục"
        self.paused = False  # chỉ dùng ở chế độ tự chạy
        self.phase = "idle"
        self.error = ""
        self.status = "running"  # running | finished | stopped | error
        self.message = ""
        self.result = "*"
        self.last = None
        self.log = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    # ---- trạng thái ----
    def active(self):
        return self.status == "running"

    def snapshot(self):
        with self.lock:
            running = self.status == "running"
            return {
                "status": self.status,
                "paused": self.paused,
                "manual": self.manual,
                "phase": self.phase,
                "error": self.error,
                "can_continue": running and (self.phase in ("ready", "error") or self.paused),
                "turn": "white" if self.board.turn == chess.WHITE else "black",
                "message": self.message,
                "result": self.result,
                "fen": self.board.fen(),
                "last": self.last,
                "white": self.players[chess.WHITE].name,
                "black": self.players[chess.BLACK].name,
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

    def _wait_go(self, release_when_auto):
        """Chờ người dùng bấm Tiếp tục (hoặc dừng ván). Nếu bật lại chế độ tự chạy thì đi luôn."""
        while not self.stop_event.is_set():
            if self.go.wait(0.2):
                return
            if release_when_auto and not self.manual:
                return

    def _think(self, player):
        """Hỏi AI nước đi. Nếu lỗi thì chờ người dùng bấm Tiếp tục để thử lại."""
        while not self.stop_event.is_set():
            with self.lock:
                self.phase = "thinking"
                self.error = ""
                board_copy = self.board.copy()
            try:
                return player.choose_move(board_copy)
            except Exception as e:  # lỗi mạng, hết quota, lỗi phía dịch vụ...
                with self.lock:
                    self.go.clear()
                    self.phase = "error"
                    self.error = f"{player.name}: {e}"
                self._wait_go(release_when_auto=False)
        return None, None

    def run(self):
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    over = self.board.is_game_over(claim_draw=True) or len(self.board.move_stack) >= self.max_plies
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
                        self.phase = "idle"
                    return
                while self.paused and not self.stop_event.is_set():
                    time.sleep(0.2)
                if self.stop_event.is_set():
                    return

                with self.lock:
                    turn = self.board.turn
                player = self.players[turn]
                move, note = self._think(player)
                if move is None or self.stop_event.is_set():
                    return

                if self.manual:  # AI đã nghĩ xong → chờ người dùng bấm Tiếp tục
                    with self.lock:
                        self.go.clear()
                        self.phase = "ready"
                    self._wait_go(release_when_auto=True)
                    if self.stop_event.is_set():
                        return

                with self.lock:
                    if move not in self.board.legal_moves:
                        self.status = "error"
                        self.message = f"{player.name} trả về nước không hợp lệ"
                        self.phase = "idle"
                        return
                    number = self.board.fullmove_number
                    san = self.board.san(move)
                    self.log.append({"n": number, "white": turn == chess.WHITE, "san": san,
                                     "who": player.name, "note": note or ""})
                    self.board.push(move)
                    self.last = move.uci()
                    self.phase = "idle"
                if not self.manual:
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
    manual: bool = True


class ModeReq(BaseModel):
    manual: bool


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.api_route("/healthz", methods=["GET", "HEAD"], response_class=PlainTextResponse)
def healthz():
    """Không cần mã truy cập: dành cho health check của Render và UptimeRobot."""
    return "ok"


# ---- danh sách model cho menu chọn ----
_models_cache = {"t": 0.0, "data": None}
_EXCLUDE = ("whisper", "tts", "guard", "embed", "imagen", "veo", "image", "aqa", "audio",
            "orpheus", "playai", "moderation", "transcribe", "live", "robotics")
PROVIDER_LABELS = {"gemini": "Gemini", "groq": "Groq", "openrouter": "OpenRouter (miễn phí)", "ollama": "Ollama (local)"}


def _list_models(provider):
    from openai import OpenAI

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


# ---- điều khiển ván cờ ----
@app.get("/api/state", dependencies=[Depends(auth)])
def state():
    if current is None:
        return {"status": "idle", "paused": False, "manual": True, "phase": "idle", "error": "",
                "can_continue": False, "turn": "white", "message": "", "result": "*",
                "fen": chess.STARTING_FEN, "last": None, "white": "", "black": "", "log": []}
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
        current = Game(white, black, delay, max_plies, req.manual)
        current.thread.start()
    return {"ok": True}


@app.post("/api/continue", dependencies=[Depends(auth)])
def continue_game():
    """Chế độ từng nước: thực hiện nước AI đã nghĩ (hoặc thử lại nếu AI gặp lỗi).
    Chế độ tự chạy: chạy tiếp sau khi tạm dừng."""
    if current is None or not current.active():
        raise HTTPException(409, "Không có ván nào đang chạy.")
    with current.lock:
        if current.phase in ("ready", "error"):
            current.go.set()
        elif current.paused:
            current.paused = False
        else:
            raise HTTPException(409, "AI chưa nghĩ xong, chưa có gì để tiếp tục.")
    return {"ok": True}


@app.post("/api/pause", dependencies=[Depends(auth)])
def pause():
    if current is None or not current.active():
        raise HTTPException(409, "Không có ván nào đang chạy.")
    with current.lock:
        if current.manual:
            raise HTTPException(409, "Ở chế độ từng nước, ván tự chờ sau mỗi nước.")
        current.paused = True
    return {"paused": True}


@app.post("/api/mode", dependencies=[Depends(auth)])
def mode(req: ModeReq):
    if current is not None and current.active():
        with current.lock:
            current.manual = req.manual
            if req.manual:
                current.paused = False
    return {"manual": req.manual}


@app.post("/api/stop", dependencies=[Depends(auth)])
def stop():
    if current is None or not current.active():
        raise HTTPException(409, "Không có ván nào đang chạy.")
    current.stop_event.set()
    with current.lock:
        current.status = "stopped"
        current.phase = "idle"
    return {"ok": True}


@app.get("/api/pgn", dependencies=[Depends(auth)], response_class=PlainTextResponse)
def pgn():
    if current is None:
        raise HTTPException(404, "Chưa có ván nào.")
    return current.pgn()
