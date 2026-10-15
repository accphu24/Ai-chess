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
import json
import os
import random
import threading
import time
import uuid

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

# ---------------------------------------------------------------------------
# Kết quả các ván đã chơi + bảng xếp hạng (Elo)
# Lưu vào data/results.json. Lưu ý: ổ đĩa Render gói miễn phí bị xóa khi deploy lại,
# nên trình duyệt cũng giữ một bản và tự đồng bộ ngược lên server.
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
RESULTS_FILE = os.path.join(DATA_DIR, "results.json")
RESULTS_LOCK = threading.RLock()
VALID_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}
MAX_RESULTS = 5000
ELO_START, ELO_K = 1000.0, 32.0


def _clean_result(r):
    try:
        res = str(r["result"])
        if res not in VALID_RESULTS:
            return None
        return {"id": str(r["id"])[:64], "ts": float(r["ts"]),
                "white": str(r["white"])[:120], "black": str(r["black"])[:120],
                "result": res, "reason": str(r.get("reason", ""))[:60], "plies": int(r.get("plies", 0)),
                "wm": int(r.get("wm", 0)), "wf": int(r.get("wf", 0)),
                "bm": int(r.get("bm", 0)), "bf": int(r.get("bf", 0))}
    except Exception:
        return None


def _load_results():
    try:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return [r for r in (_clean_result(x) for x in data) if r]
    except Exception:
        return []


def _save_results():  # gọi khi đang giữ RESULTS_LOCK
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = RESULTS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False)
        os.replace(tmp, RESULTS_FILE)
    except Exception:
        pass  # ổ đĩa chỉ đọc: bỏ qua, trình duyệt vẫn giữ bản sao


RESULTS = _load_results()

# --- MongoDB (tùy chọn): đặt MONGODB_URI để bật. Không có/không kết nối được thì dùng file. ---
# Biến môi trường: MONGODB_URI (bắt buộc để bật), MONGODB_DB (mặc định ai_projects),
# MONGODB_COLLECTION (mặc định chess_arena_results - collection riêng nên không đụng dự án khác)
_mongo_state = {"coll": None, "fail_t": 0.0, "error": ""}


def _mongo_fail(e):
    _mongo_state.update(coll=None, fail_t=time.time(), error=type(e).__name__)  # không ghi chi tiết lỗi để khỏi lộ thông tin


def _sync_union(coll):
    """Hợp nhất 2 chiều: kết quả trong MongoDB <-> kết quả đang có trong bộ nhớ."""
    from pymongo import UpdateOne

    docs = [r for r in (_clean_result(d) for d in coll.find({}, {"_id": 0})) if r]
    mongo_ids = {d["id"] for d in docs}
    with RESULTS_LOCK:
        local_ids = {r["id"] for r in RESULTS}
        for d in docs:
            if d["id"] not in local_ids and len(RESULTS) < MAX_RESULTS:
                RESULTS.append(d)
        missing = [r for r in RESULTS if r["id"] not in mongo_ids]
        _save_results()
    if missing:
        coll.bulk_write([UpdateOne({"id": r["id"]}, {"$set": r}, upsert=True) for r in missing], ordered=False)


def _mongo():
    uri = os.environ.get("MONGODB_URI", "").strip()
    st = _mongo_state
    if not uri:
        return None
    if st["coll"] is not None:
        return st["coll"]
    if time.time() - st["fail_t"] < 60:  # vừa lỗi: đừng thử lại liên tục
        return None
    try:
        import pymongo

        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        coll = client[os.environ.get("MONGODB_DB", "ai_projects")][
            os.environ.get("MONGODB_COLLECTION", "chess_arena_results")]
        coll.create_index("id", unique=True)  # cũng là bước kiểm tra kết nối
        _sync_union(coll)
        st.update(coll=coll, error="")
        return coll
    except Exception as e:
        _mongo_fail(e)
        return None


def storage_info():
    coll = _mongo()
    configured = bool(os.environ.get("MONGODB_URI", "").strip())
    return {"backend": "mongodb" if coll is not None else "file", "configured": configured,
            "error": _mongo_state["error"] if configured and coll is None else ""}


def _persist_upsert(recs):
    coll = _mongo()
    if coll is None or not recs:
        return
    try:
        from pymongo import UpdateOne

        coll.bulk_write([UpdateOne({"id": r["id"]}, {"$set": r}, upsert=True) for r in recs], ordered=False)
    except Exception as e:
        _mongo_fail(e)


def add_result(rec):
    with RESULTS_LOCK:
        RESULTS.append(rec)
        _save_results()
    _persist_upsert([rec])


def merge_results(items):
    new = []
    with RESULTS_LOCK:
        ids = {r["id"] for r in RESULTS}
        for item in items:
            r = _clean_result(item)
            if r and r["id"] not in ids and len(RESULTS) < MAX_RESULTS:
                RESULTS.append(r)
                ids.add(r["id"])
                new.append(r)
        if new:
            _save_results()
    _persist_upsert(new)
    return len(new)


def clear_all_results():
    with RESULTS_LOCK:
        RESULTS.clear()
        _save_results()
    coll = _mongo()
    if coll is not None:
        try:
            coll.delete_many({})
        except Exception as e:
            _mongo_fail(e)


def compute_leaderboard(results):
    """Elo tính lại từ đầu theo thứ tự thời gian. Ván cùng một model hai bên không được tính."""
    stats = {}

    def get(name):
        return stats.setdefault(name, {"name": name, "elo": ELO_START, "games": 0, "w": 0, "d": 0, "l": 0,
                                       "points": 0.0, "moves": 0, "fallbacks": 0})

    for r in sorted(results, key=lambda x: x["ts"]):
        if r["white"] == r["black"]:
            continue
        w, b = get(r["white"]), get(r["black"])
        sw = 1.0 if r["result"] == "1-0" else 0.0 if r["result"] == "0-1" else 0.5  # "*" (hết giới hạn nước) = hòa
        exp_w = 1.0 / (1.0 + 10 ** ((b["elo"] - w["elo"]) / 400.0))
        w["elo"] += ELO_K * (sw - exp_w)
        b["elo"] += ELO_K * ((1.0 - sw) - (1.0 - exp_w))
        for side, s in ((w, sw), (b, 1.0 - sw)):
            side["games"] += 1
            side["points"] += s
            side["w" if s == 1.0 else "l" if s == 0.0 else "d"] += 1
        w["moves"] += r["wm"]; w["fallbacks"] += r["wf"]
        b["moves"] += r["bm"]; b["fallbacks"] += r["bf"]

    rows = []
    for s in stats.values():
        rows.append({"name": s["name"], "elo": round(s["elo"]), "games": s["games"],
                     "w": s["w"], "d": s["d"], "l": s["l"],
                     "score_pct": round(100.0 * s["points"] / s["games"], 1) if s["games"] else 0.0,
                     "illegal_pct": round(100.0 * s["fallbacks"] / s["moves"], 1) if s["moves"] else 0.0})
    rows.sort(key=lambda x: (-x["elo"], -x["score_pct"], x["name"]))
    return rows



class Game:
    """phase: idle (giữa 2 nước) | thinking (AI đang nghĩ) | ready (AI đã nghĩ xong, chờ bấm Tiếp tục)
    | error (AI gặp lỗi, chờ bấm Tiếp tục để thử lại)."""

    def __init__(self, white, black, delay, max_plies, manual, auto_retries=None):
        self.players = {chess.WHITE: white, chess.BLACK: black}
        self.delay = delay
        self.max_plies = max_plies
        self.manual = manual
        self.auto_retries = auto_retries  # None = chờ người bấm Tiếp tục khi lỗi; số = tự thử lại rồi hủy ván
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
        self.moves = {chess.WHITE: 0, chess.BLACK: 0}
        self.fallbacks = {chess.WHITE: 0, chess.BLACK: 0}
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

    def _record(self):  # gọi khi đang giữ self.lock
        return {"id": uuid.uuid4().hex, "ts": time.time(),
                "white": self.players[chess.WHITE].name, "black": self.players[chess.BLACK].name,
                "result": self.result, "reason": self.message, "plies": len(self.board.move_stack),
                "wm": self.moves[chess.WHITE], "wf": self.fallbacks[chess.WHITE],
                "bm": self.moves[chess.BLACK], "bf": self.fallbacks[chess.BLACK]}

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

    def stop(self):
        self.stop_event.set()
        with self.lock:
            if self.status == "running":
                self.status = "stopped"
                self.phase = "idle"

    def _think(self, player):
        """Hỏi AI nước đi. Nếu lỗi thì chờ người dùng bấm Tiếp tục để thử lại
        (hoặc, ở giải đấu, tự thử lại vài lần rồi hủy ván)."""
        errors = 0
        while not self.stop_event.is_set():
            with self.lock:
                self.phase = "thinking"
                self.error = ""
                board_copy = self.board.copy()
            try:
                return player.choose_move(board_copy)
            except Exception as e:  # lỗi mạng, hết quota, lỗi phía dịch vụ...
                errors += 1
                if self.auto_retries is not None:
                    if errors > self.auto_retries:
                        with self.lock:
                            self.status = "error"
                            self.message = f"{player.name}: {e}"
                            self.phase = "idle"
                        return None, None
                    self._sleep(TOURNEY_RETRY_SLEEP * errors)
                    continue
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
                        rec = self._record()
                    add_result(rec)  # ngoài khóa: lưu MongoDB có thể mất vài giây
                    with self.lock:
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
                    self.moves[turn] += 1
                    if (note or "").startswith("⚠"):
                        self.fallbacks[turn] += 1
                    self.last = move.uci()
                    self.phase = "idle"
                if not self.manual:
                    self._sleep(self.delay)
        finally:
            for p in self.players.values():
                p.close()


current = None
tournament = None
GLOCK = threading.Lock()

MAX_TOURNEY_PLAYERS = 8
MAX_TOURNEY_GAMES = 60
TOURNEY_RETRY_SLEEP = 10.0  # giây; lần lỗi thứ k chờ k*10 giây rồi thử lại


@app.on_event("startup")
def _connect_storage():
    threading.Thread(target=_mongo, daemon=True).start()


class StartReq(BaseModel):
    white: str = "random"
    black: str = "random"
    delay: float = 1.0
    max_plies: int = 200
    manual: bool = True


class ModeReq(BaseModel):
    manual: bool


class Tournament:
    """Giải vòng tròn: mỗi cặp đấu games_per_pair ván, đổi màu xen kẽ, chạy tự động lần lượt từng ván."""

    def __init__(self, specs, names, games_per_pair, delay, max_plies):
        self.specs, self.names = specs, names
        self.delay, self.max_plies = delay, max_plies
        pairs = [(i, j) for i in range(len(specs)) for j in range(i + 1, len(specs))]
        self.schedule = []
        for g in range(games_per_pair):
            for i, j in pairs:
                self.schedule.append((specs[i], specs[j]) if g % 2 == 0 else (specs[j], specs[i]))
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.status = "running"  # running | finished | stopped
        self.index = 0
        self.games = []
        self.game = None
        self.stats = {n: {"name": n, "points": 0.0, "games": 0, "w": 0, "d": 0, "l": 0} for n in names.values()}
        self.thread = threading.Thread(target=self.run, daemon=True)

    def active(self):
        return self.status == "running"

    def _set(self, entry, **kw):
        with self.lock:
            entry.update(kw)

    def _score(self, white, black, result):
        sw = 1.0 if result == "1-0" else 0.0 if result == "0-1" else 0.5  # "*" (hết giới hạn nước) = hòa
        for name, s in ((white, sw), (black, 1.0 - sw)):
            st = self.stats[name]
            st["games"] += 1
            st["points"] += s
            st["w" if s == 1.0 else "l" if s == 0.0 else "d"] += 1

    def stop(self):
        self.stop_event.set()
        g = self.game
        if g is not None:
            g.stop()

    def snapshot(self):
        with self.lock:
            standings = sorted(self.stats.values(), key=lambda s: (-s["points"], -s["w"], s["name"]))
            return {"status": self.status, "active": self.active(), "index": self.index,
                    "total": len(self.schedule), "games": [dict(g) for g in self.games],
                    "standings": [dict(s) for s in standings]}

    def run(self):
        global current
        try:
            for idx, (ws, bs) in enumerate(self.schedule):
                if self.stop_event.is_set():
                    break
                entry = {"white": self.names[ws], "black": self.names[bs], "result": "", "reason": ""}
                with self.lock:
                    self.index = idx + 1
                    self.games.append(entry)
                try:
                    white = make_player(ws)
                except (Exception, SystemExit) as e:
                    self._set(entry, result="void", reason=f"Không tạo được người chơi ({type(e).__name__})")
                    continue
                try:
                    black = make_player(bs)
                except (Exception, SystemExit) as e:
                    white.close()
                    self._set(entry, result="void", reason=f"Không tạo được người chơi ({type(e).__name__})")
                    continue
                game = Game(white, black, self.delay, self.max_plies, False, auto_retries=3)
                with GLOCK:
                    current = game
                with self.lock:
                    self.game = game
                game.thread.start()
                while game.thread.is_alive():
                    game.thread.join(0.5)
                    if self.stop_event.is_set():
                        break
                with self.lock:
                    if game.status == "finished":
                        entry.update(result=game.result, reason=game.message)
                        self._score(entry["white"], entry["black"], game.result)
                    elif self.stop_event.is_set():
                        entry.update(result="stopped", reason="Đã dừng")
                    else:  # lỗi API quá nhiều lần: hủy ván, không tính điểm
                        entry.update(result="void", reason=(game.message or "Ván bị hủy")[:120])
                if not self.stop_event.is_set():
                    time.sleep(2)  # nghỉ giữa các ván cho dịu API
        finally:
            with self.lock:
                self.status = "stopped" if self.stop_event.is_set() else "finished"
                self.game = None


class TournamentReq(BaseModel):
    models: list[str] = []
    games_per_pair: int = 2
    delay: float = 0.5
    max_plies: int = 200


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
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=8 if key_env else 1.5, max_retries=0)
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
    if any(g["items"] for g in groups[1:]):  # chỉ hiện khi có ít nhất 1 model AI để bốc
        groups.insert(1, {"label": "Đặc biệt", "items": [
            {"spec": "pick:random", "label": "🎲 Ngẫu nhiên một model"}]})
    data = {"groups": groups, "notes": notes}
    _models_cache.update(t=now, data=data)
    return data


def _pool():
    """Các model AI để bốc ngẫu nhiên (không gồm Random/Stockfish)."""
    data = models(refresh=False)
    return [it["spec"] for g in data["groups"] if g["label"] not in ("Cơ bản", "Đặc biệt") for it in g["items"]]


def _resolve_specs(white, black):
    """Thay 'pick:random' bằng một model thật; cố gắng không để hai bên trùng nhau."""
    white, black = white.strip(), black.strip()
    if "pick:random" not in (white, black):
        return white, black
    pool = _pool()
    if not pool:
        raise HTTPException(400, "Chưa có model AI nào để bốc ngẫu nhiên (cần đặt API key của ít nhất một dịch vụ).")
    if white == "pick:random":
        white = random.choice([s for s in pool if s != black] or pool)
    if black == "pick:random":
        black = random.choice([s for s in pool if s != white] or pool)
    return white, black


# ---- điều khiển ván cờ ----
@app.get("/api/state", dependencies=[Depends(auth)])
def state():
    if current is None:
        return {"tournament": tournament.snapshot() if tournament else None,
                "results_count": len(RESULTS), "status": "idle", "paused": False, "manual": True, "phase": "idle", "error": "",
                "can_continue": False, "turn": "white", "message": "", "result": "*",
                "fen": chess.STARTING_FEN, "last": None, "white": "", "black": "", "log": []}
    data = current.snapshot()
    data["results_count"] = len(RESULTS)
    data["tournament"] = tournament.snapshot() if tournament else None
    return data


@app.post("/api/start", dependencies=[Depends(auth)])
def start(req: StartReq):
    global current
    with GLOCK:
        if (current and current.active()) or (tournament and tournament.active()):
            raise HTTPException(409, "Đang có ván hoặc giải đấu chạy. Hãy dừng trước.")
        white_spec, black_spec = _resolve_specs(req.white, req.black)
        try:
            white = make_player(white_spec)
        except (Exception, SystemExit) as e:
            raise HTTPException(400, f"Không tạo được người chơi Trắng: {e}")
        try:
            black = make_player(black_spec)
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
    if tournament is not None and tournament.active():
        tournament.stop()
        return {"ok": True}
    if current is None or not current.active():
        raise HTTPException(409, "Không có ván nào đang chạy.")
    current.stop()
    return {"ok": True}


@app.get("/api/pgn", dependencies=[Depends(auth)], response_class=PlainTextResponse)
def pgn():
    if current is None:
        raise HTTPException(404, "Chưa có ván nào.")
    return current.pgn()


# ---- bảng xếp hạng ----
class ResultsIn(BaseModel):
    results: list = []


@app.get("/api/leaderboard", dependencies=[Depends(auth)])
def leaderboard():
    with RESULTS_LOCK:
        snapshot = list(RESULTS)
    recent = sorted(snapshot, key=lambda r: r["ts"], reverse=True)[:15]
    return {"leaderboard": compute_leaderboard(snapshot), "recent": recent, "total": len(snapshot),
            "storage": storage_info()}


@app.get("/api/results", dependencies=[Depends(auth)])
def get_results():
    with RESULTS_LOCK:
        return {"results": list(RESULTS)}


@app.post("/api/results/import", dependencies=[Depends(auth)])
def import_results(req: ResultsIn):
    return {"added": merge_results(req.results[:MAX_RESULTS])}


@app.delete("/api/results", dependencies=[Depends(auth)])
def clear_results():
    clear_all_results()
    return {"ok": True}


# ---- giải đấu ----
@app.post("/api/tournament/start", dependencies=[Depends(auth)])
def tournament_start(req: TournamentReq):
    global tournament
    specs = list(dict.fromkeys(s.strip() for s in req.models if s.strip()))
    if any(s.startswith("pick:") for s in specs):
        raise HTTPException(400, "Giải đấu cần chọn model cụ thể, không dùng 'Ngẫu nhiên'.")
    if len(specs) < 2:
        raise HTTPException(400, "Cần chọn ít nhất 2 người chơi.")
    if len(specs) > MAX_TOURNEY_PLAYERS:
        raise HTTPException(400, f"Tối đa {MAX_TOURNEY_PLAYERS} người chơi mỗi giải.")
    gpp = max(1, min(4, req.games_per_pair))
    total = len(specs) * (len(specs) - 1) // 2 * gpp
    if total > MAX_TOURNEY_GAMES:
        raise HTTPException(400, f"Giải có {total} ván, vượt giới hạn {MAX_TOURNEY_GAMES}. Hãy bớt model hoặc giảm số ván mỗi cặp.")
    with GLOCK:
        if (current and current.active()) or (tournament and tournament.active()):
            raise HTTPException(409, "Đang có ván hoặc giải đấu chạy. Hãy dừng trước.")
        names = {}
        for s in specs:  # tạo thử để báo lỗi sớm (thiếu key, sai tên...) và lấy tên hiển thị
            try:
                pl = make_player(s)
            except (Exception, SystemExit) as e:
                raise HTTPException(400, f"Không tạo được '{s}': {e}")
            names[s] = pl.name
            pl.close()
        if len(set(names.values())) < len(specs):
            raise HTTPException(400, "Có hai lựa chọn trùng tên hiển thị.")
        tournament = Tournament(specs, names, gpp, max(0.0, min(30.0, req.delay)),
                                max(10, min(400, req.max_plies)))
        tournament.thread.start()
    return {"ok": True, "total": total}
