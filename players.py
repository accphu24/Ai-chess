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
    "You will be given the current position (FEN), the move history, and the "
    "list of all legal moves in UCI notation. Think briefly (at most 3 short "
    "sentences), then finish with a final line in exactly this format:\n"
    "MOVE: <uci move>\n"
    "The move MUST be one of the legal moves listed. Example: MOVE: e2e4"
)

MOVE_RE = re.compile(r"MOVE:\s*([a-h][1-8][a-h][1-8][qrbn]?)", re.IGNORECASE)


def san_history(board: chess.Board) -> str:
    b = chess.Board()
    parts = []
    for m in board.move_stack:
        if b.turn == chess.WHITE:
            parts.append(f"{b.fullmove_number}.")
        parts.append(b.san(m))
        b.push(m)
    return " ".join(parts) if parts else "(chưa có nước đi)"


class LLMPlayer(Player):
    """Lớp chung cho mọi model ngôn ngữ; lớp con chỉ cần cài _complete()."""

    def __init__(self, model, max_retries=3):
        self.model = model
        self.name = model
        self.max_retries = max_retries

    def _complete(self, messages) -> str:
        raise NotImplementedError

    def choose_move(self, board):
        side = "White" if board.turn == chess.WHITE else "Black"
        legal = [m.uci() for m in board.legal_moves]
        prompt = (
            f"You are playing {side}.\n"
            f"FEN: {board.fen()}\n"
            f"Moves so far: {san_history(board)}\n"
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
                    return chess.Move.from_uci(uci), note
                feedback = f"'{uci}' is not a legal move. Legal moves: {' '.join(legal)}. "
            else:
                feedback = "I couldn't find a 'MOVE: <uci>' line. "
            messages.append({"role": "assistant", "content": text or "(empty)"})
            messages.append({"role": "user", "content": feedback + "Reply again with a legal move."})

        # Hết lượt thử: đi ngẫu nhiên để ván cờ không bị kẹt, và ghi rõ trong log
        fallback = random.choice(list(board.legal_moves))
        return fallback, f"⚠ model trả nước sai {self.max_retries} lần → đi ngẫu nhiên. Trả lời cuối: {last_text[:120]}"


class ClaudePlayer(LLMPlayer):
    def __init__(self, model, max_retries=3):
        import anthropic  # import muộn để các chế độ khác không cần SDK

        super().__init__(model, max_retries)
        self.client = anthropic.Anthropic()  # đọc ANTHROPIC_API_KEY từ môi trường

    def _complete(self, messages):
        resp = self.client.messages.create(
            model=self.model, max_tokens=400, system=SYSTEM_PROMPT, messages=messages
        )
        return "".join(b.text for b in resp.content if b.type == "text")


# Các dịch vụ có API tương thích OpenAI (nhiều nơi có gói miễn phí / chạy local)
PRESETS = {
    "ollama": ("http://localhost:11434/v1", None),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
}


class OpenAICompatPlayer(LLMPlayer):
    def __init__(self, provider, model):
        from openai import OpenAI

        base_url, key_env = PRESETS[provider]
        api_key = os.environ.get(key_env, "") if key_env else "ollama"
        if key_env and not api_key:
            raise SystemExit(f"Thiếu biến môi trường {key_env} cho '{provider}'")
        super().__init__(model, max_retries=5)  # model nhỏ hay đi sai nên cho thử nhiều hơn
        self.name = f"{model} ({provider})"
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _complete(self, messages):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        for attempt in range(3):  # gói miễn phí hay bị giới hạn tốc độ → chờ rồi thử lại
            try:
                r = self.client.chat.completions.create(model=self.model, messages=msgs, max_tokens=600)
                text = r.choices[0].message.content or ""
                return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(8 * (attempt + 1))


def make_player(spec: str) -> Player:
    kind, _, arg = spec.partition(":")
    kind = kind.lower()
    if kind in PRESETS:
        if not arg:
            raise SystemExit(f"Cần tên model, ví dụ {kind}:<model-id>")
        return OpenAICompatPlayer(kind, arg)
    if kind == "claude":
        return ClaudePlayer(arg or "claude-sonnet-5-5")
    if kind == "stockfish":
        return StockfishPlayer(arg or "stockfish")
    if kind == "random":
        return RandomPlayer()
    raise SystemExit(f"Loại người chơi không hợp lệ: {spec}")
