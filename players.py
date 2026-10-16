"""Các loại người chơi (Claude, Stockfish, Gemini/Groq/OpenRouter/Ollama, random).
Dùng chung cho giao diện desktop (arena.py) và web (server.py)."""

import os
import random
import re
import time

import chess
import chess.engine

# --------------------------------------------------------------------------
# Người chơi
# --------------------------------------------------------------------------


class Player:
    name = "Player"

    def choose_move(self, board: chess.Board):
        """Trả về (move, ghi_chú). move=None nghĩa là không đi được."""
        raise NotImplementedError

    def close(self):
        pass

    def usage(self):
        """(token vào, token ra) đã dùng; chỉ model AI mới có."""
        return (0, 0)


class QuotaExhausted(Exception):
    """Hết hạn mức (quota) trong ngày hoặc hết credit: thử lại ngay cũng vô ích."""


class RandomPlayer(Player):
    name = "Random"

    def choose_move(self, board):
        return random.choice(list(board.legal_moves)), "ngẫu nhiên"


class StockfishPlayer(Player):
    def __init__(self, path="stockfish", think_time=0.3):
        self.name = "Stockfish"
        self.think_time = think_time
        self.engine = chess.engine.SimpleEngine.popen_uci(path)

    def choose_move(self, board):
        result = self.engine.play(board, chess.engine.Limit(time=self.think_time))
        return result.move, f"engine ({self.think_time}s)"

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


SYSTEM_PROMPT = (
    "You are a strong chess player playing a game against another AI. "
    "You will be given the current position (FEN), the most recent moves, and the "
    "list of all legal moves in UCI notation. Think briefly (at most 3 short "
    "sentences), then finish with a final line in exactly this format:\n"
    "MOVE: <uci move>\n"
    "The move MUST be one of the legal moves listed. Example: MOVE: e2e4"
)

# Chế độ tiết kiệm token: không giải thích, chỉ trả đúng một dòng
SYSTEM_PROMPT_BRIEF = (
    "You are a strong chess player. You will be given the current position (FEN), the most "
    "recent moves, and the list of legal moves in UCI notation. Reply with ONLY one line, "
    "with no explanation and no other text, in exactly this format:\n"
    "MOVE: <uci move>\n"
    "The move MUST be one of the legal moves listed. Example: MOVE: e2e4"
)

MOVE_RE = re.compile(r"MOVE:\s*([a-h][1-8][a-h][1-8][qrbn]?)", re.IGNORECASE)
HISTORY_PLIES = 16  # chỉ gửi các nước gần nhất (FEN đã mô tả đủ thế cờ) để đỡ tốn token


def san_history(board: chess.Board, max_plies: int = HISTORY_PLIES) -> str:
    b = chess.Board()
    items = []
    for m in board.move_stack:
        items.append((b.fullmove_number, b.turn == chess.WHITE, b.san(m)))
        b.push(m)
    if not items:
        return "(chưa có nước đi)"
    cut = len(items) > max_plies
    items = items[-max_plies:]
    parts = []
    for k, (num, white, san) in enumerate(items):
        if white:
            parts.append(f"{num}.")
        elif k == 0:
            parts.append(f"{num}...")
        parts.append(san)
    return ("... " if cut else "") + " ".join(parts)


def _is_daily_quota(e) -> bool:
    low = str(e).lower()
    return any(k in low for k in ("per day", "perday", "per-day", "daily", "insufficient_quota"))


def _retry_after(e):
    """Số giây dịch vụ yêu cầu chờ (header Retry-After hoặc trong nội dung lỗi), nếu có."""
    try:
        h = e.response.headers.get("retry-after")
        if h:
            return float(h)
    except Exception:
        pass
    m = re.search(r"retry(?:Delay)?\W+(?:in\s+)?([\d.]+)\s*s", str(e), re.IGNORECASE)
    return float(m.group(1)) if m else None


class LLMPlayer(Player):
    """Lớp chung cho mọi model ngôn ngữ; lớp con chỉ cần cài _complete()."""

    def __init__(self, model, max_retries=3, brief=False):
        self.model = model
        self.name = model
        self.max_retries = max_retries
        self.brief = brief
        self.tokens_in = 0
        self.tokens_out = 0

    def usage(self):
        return (self.tokens_in, self.tokens_out)

    def _add_usage(self, tin, tout):
        self.tokens_in += int(tin or 0)
        self.tokens_out += int(tout or 0)

    def _system(self):
        return SYSTEM_PROMPT_BRIEF if self.brief else SYSTEM_PROMPT

    def _complete(self, messages) -> str:
        raise NotImplementedError

    def choose_move(self, board):
        side = "White" if board.turn == chess.WHITE else "Black"
        legal = [m.uci() for m in board.legal_moves]
        prompt = (
            f"You are playing {side}.\n"
            f"FEN: {board.fen()}\n"
            f"Recent moves: {san_history(board)}\n"
            f"Legal moves (UCI): {' '.join(legal)}\n"
            "Choose your move."
        )
        messages = [{"role": "user", "content": prompt}]
        last_text = ""

        for attempt in range(1, self.max_retries + 1):
            text = self._complete(messages).strip()
            last_text = text
            found = MOVE_RE.findall(text)
            if found:
                uci = found[-1].lower()
                if uci in legal:
                    comment = MOVE_RE.sub("", text).strip()
                    note = comment[:300] + (f" [thử lần {attempt}]" if attempt > 1 else "")
                    return chess.Move.from_uci(uci), note.strip()
                feedback = f"'{uci}' is not a legal move. Legal moves: {' '.join(legal)}. "
            else:
                feedback = "I couldn't find a 'MOVE: <uci>' line. "
            messages.append({"role": "assistant", "content": text or "(empty)"})
            messages.append({"role": "user", "content": feedback + "Reply again with a legal move."})

        # Hết lượt thử: đi ngẫu nhiên để ván cờ không bị kẹt, và ghi rõ trong log
        fallback = random.choice(list(board.legal_moves))
        return fallback, f"⚠ model trả nước sai {self.max_retries} lần → đi ngẫu nhiên. Trả lời cuối: {last_text[:120]}"


class ClaudePlayer(LLMPlayer):
    def __init__(self, model, max_retries=3, brief=False):
        import anthropic  # import muộn để các chế độ khác không cần SDK

        super().__init__(model, max_retries, brief)
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=4)  # đọc ANTHROPIC_API_KEY từ môi trường

    def _complete(self, messages):
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=100 if self.brief else 400,
                system=self._system(), messages=messages,
            )
        except self.anthropic.BadRequestError as e:
            if "credit balance" in str(e).lower():
                raise QuotaExhausted("Claude: tài khoản Anthropic đã hết credit.") from e
            raise
        u = getattr(resp, "usage", None)
        if u:
            self._add_usage(getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))
        return "".join(b.text for b in resp.content if b.type == "text")


# Các dịch vụ có API tương thích OpenAI (nhiều nơi có gói miễn phí / chạy local)
PRESETS = {
    "ollama": ("http://localhost:11434/v1", None),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
}


class OpenAICompatPlayer(LLMPlayer):
    def __init__(self, provider, model, brief=False):
        import openai

        base_url, key_env = PRESETS[provider]
        api_key = os.environ.get(key_env, "") if key_env else "ollama"
        if key_env and not api_key:
            raise SystemExit(f"Thiếu biến môi trường {key_env} cho '{provider}'")
        super().__init__(model, max_retries=5, brief=brief)  # model nhỏ hay đi sai nên cho thử nhiều hơn
        self.provider = provider
        self.name = f"{model} ({provider})"
        self.openai = openai
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0)  # thử lại do code bên dưới tự xử lý
        self.max_tokens = 300 if brief else 600
        self.use_effort = provider != "ollama"  # xin model suy luận ở mức thấp để bớt tốn token

    def _complete(self, messages):
        oa = self.openai
        msgs = [{"role": "system", "content": self._system()}] + messages
        for attempt in range(5):
            kwargs = dict(model=self.model, messages=msgs, max_tokens=self.max_tokens)
            if self.use_effort:
                kwargs["reasoning_effort"] = "low"
            try:
                r = self.client.chat.completions.create(**kwargs)
            except oa.BadRequestError:
                if self.use_effort:  # model không hỗ trợ tham số này → bỏ đi rồi thử lại
                    self.use_effort = False
                    continue
                raise
            except oa.RateLimitError as e:
                if _is_daily_quota(e):
                    raise QuotaExhausted(
                        f"{self.provider}: đã hết hạn mức trong ngày ({self.model}). "
                        "Hãy thử model khác hoặc chờ sang ngày mới.") from e
                if attempt == 4:
                    raise
                time.sleep(min(_retry_after(e) or 15 * (attempt + 1), 60))
                continue
            except (oa.APIConnectionError, oa.InternalServerError):
                if attempt == 4:
                    raise
                time.sleep(6 * (attempt + 1))
                continue
            u = getattr(r, "usage", None)
            if u:
                self._add_usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
            choice = r.choices[0]
            text = choice.message.content or ""
            if choice.finish_reason == "length":  # bị cắt (thường do model suy luận dài) → nới giới hạn cho lần sau
                self.max_tokens = min(self.max_tokens * 2, 2400)
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        raise RuntimeError("Không gọi được API sau nhiều lần thử")


def make_player(spec: str, brief: bool = False) -> Player:
    kind, _, arg = spec.partition(":")
    kind = kind.lower()
    if kind in PRESETS:
        if not arg:
            raise SystemExit(f"Cần tên model, ví dụ {kind}:<model-id>")
        return OpenAICompatPlayer(kind, arg, brief=brief)
    if kind == "claude":
        return ClaudePlayer(arg or "claude-sonnet-5-5", brief=brief)
    if kind == "stockfish":
        return StockfishPlayer(arg or "stockfish")
    if kind == "random":
        return RandomPlayer()
    raise SystemExit(f"Loại người chơi không hợp lệ: {spec}")
