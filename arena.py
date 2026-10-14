#!/usr/bin/env python3
"""
Chess AI Arena - xem hai AI chơi cờ vua với nhau (Python + tkinter).

Cài đặt:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY="sk-ant-..."      # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
    (Linux có thể cần: sudo apt install python3-tk)

Chạy:
    python arena.py                                   # Sonnet (trắng) vs Haiku (đen)
    python arena.py --white claude:claude-opus-5-5 --black claude:claude-haiku-5-5
    python arena.py --white claude:claude-sonnet-5-5 --black stockfish
    python arena.py --white stockfish:/usr/games/stockfish --black random

Loại người chơi (--white / --black):
    claude:<model-id>      gọi Anthropic API
    stockfish[:<đường dẫn>] engine Stockfish (cần cài sẵn)
    random                 đi nước ngẫu nhiên (để thử nghiệm, không tốn tiền)
    ollama:<model>         model chạy local qua Ollama (miễn phí hoàn toàn)
    groq:<model>           Groq API (cần GROQ_API_KEY)
    gemini:<model>         Google Gemini API (cần GEMINI_API_KEY)
    openrouter:<model>     OpenRouter (cần OPENROUTER_API_KEY)

Muốn thêm nhà cung cấp khác (OpenAI, Gemini, model local...): viết một class
kế thừa `Player`, cài đặt `choose_move(board)` rồi đăng ký trong `make_player`.
"""

import argparse
import datetime
import os
import queue
import random
import re
import threading
import time
import tkinter as tk
from tkinter import ttk

import chess
import chess.engine
import chess.pgn

from players import Player, make_player

# --------------------------------------------------------------------------
# Giao diện
# --------------------------------------------------------------------------

GLYPHS = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}
LIGHT, DARK = "#f0d9b5", "#b58863"
HILITE = "#f7ec74"
SQ = 64


class App:
    def __init__(self, root, white: Player, black: Player, delay: float, max_plies: int):
        self.root = root
        self.players = {chess.WHITE: white, chess.BLACK: black}
        self.delay_ms = int(delay * 1000)
        self.max_plies = max_plies
        self.board = chess.Board()
        self.q = queue.Queue()
        self.thinking = False
        self.paused = False
        self.finished = False
        self.last_move = None

        root.title(f"Chess AI Arena — {white.name} (trắng) vs {black.name} (đen)")
        left = ttk.Frame(root, padding=8)
        left.grid(row=0, column=0)
        right = ttk.Frame(root, padding=8)
        right.grid(row=0, column=1, sticky="ns")

        self.canvas = tk.Canvas(left, width=SQ * 8 + 20, height=SQ * 8 + 20, highlightthickness=0)
        self.canvas.pack()

        self.status = ttk.Label(right, text="", font=("TkDefaultFont", 11, "bold"), wraplength=320)
        self.status.pack(anchor="w", pady=(0, 6))

        btns = ttk.Frame(right)
        btns.pack(anchor="w", pady=(0, 6))
        self.pause_btn = ttk.Button(btns, text="Tạm dừng", command=self.toggle_pause)
        self.pause_btn.pack(side="left", padx=2)
        ttk.Button(btns, text="Đi 1 nước", command=self.step).pack(side="left", padx=2)

        frame = ttk.Frame(right)
        frame.pack(fill="both", expand=True)
        self.log = tk.Text(frame, width=48, height=30, wrap="word", state="disabled")
        sb = ttk.Scrollbar(frame, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.log.tag_configure("move", font=("TkDefaultFont", 10, "bold"))
        self.log.tag_configure("note", foreground="#555555")

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.draw()
        self.root.after(100, self.poll)
        self.root.after(500, self.next_move)

    # ---- vẽ bàn cờ ----
    def draw(self):
        c = self.canvas
        c.delete("all")
        off = 10
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, 7 - rank)
                color = LIGHT if (file + rank) % 2 == 0 else DARK
                if self.last_move and sq in (self.last_move.from_square, self.last_move.to_square):
                    color = HILITE
                x, y = off + file * SQ, off + rank * SQ
                c.create_rectangle(x, y, x + SQ, y + SQ, fill=color, outline="")
                piece = self.board.piece_at(sq)
                if piece:
                    c.create_text(x + SQ / 2, y + SQ / 2, text=GLYPHS[piece.symbol()],
                                  font=("DejaVu Sans", 38), fill="#000000")
        for i in range(8):
            c.create_text(off + i * SQ + SQ / 2, off + 8 * SQ + 8, text="abcdefgh"[i], font=("TkDefaultFont", 9))
            c.create_text(4, off + i * SQ + SQ / 2, text=str(8 - i), font=("TkDefaultFont", 9))

    def write_log(self, text, tag=None):
        self.log.configure(state="normal")
        self.log.insert("end", text, tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---- điều khiển ----
    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.configure(text="Tiếp tục" if self.paused else "Tạm dừng")
        if not self.paused:
            self.next_move()
        else:
            self.status.configure(text="Đã tạm dừng")

    def step(self):
        if self.paused:
            self.next_move(force=True)

    def next_move(self, force=False):
        if self.finished or self.thinking or (self.paused and not force):
            return
        if self.board.is_game_over(claim_draw=True) or len(self.board.move_stack) >= self.max_plies:
            return self.finish()
        player = self.players[self.board.turn]
        side = "Trắng" if self.board.turn == chess.WHITE else "Đen"
        self.status.configure(text=f"{side} ({player.name}) đang suy nghĩ...")
        self.thinking = True
        board_copy = self.board.copy()

        def work():
            try:
                self.q.put(player.choose_move(board_copy))
            except Exception as e:  # lỗi mạng, hết quota, ...
                self.q.put((None, f"Lỗi: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            move, note = self.q.get_nowait()
        except queue.Empty:
            pass
        else:
            self.thinking = False
            self.apply(move, note)
        self.root.after(100, self.poll)

    def apply(self, move, note):
        if move is None or move not in self.board.legal_moves:
            self.paused = True
            self.pause_btn.configure(text="Tiếp tục")
            self.status.configure(text="Có lỗi, đã tạm dừng. Bấm 'Tiếp tục' để thử lại.")
            self.write_log(f"{note}\n", "note")
            return
        number = self.board.fullmove_number
        is_white = self.board.turn == chess.WHITE
        san = self.board.san(move)
        who = self.players[self.board.turn].name
        self.board.push(move)
        self.last_move = move
        prefix = f"{number}. " if is_white else f"{number}... "
        self.write_log(f"{prefix}{san}  ({who})\n", "move")
        if note:
            self.write_log(f"   {note}\n", "note")
        self.draw()
        self.root.after(self.delay_ms, self.next_move)

    def finish(self):
        self.finished = True
        outcome = self.board.outcome(claim_draw=True)
        if outcome:
            result = outcome.result()
            reason = outcome.termination.name
        else:
            result, reason = "*", "đạt giới hạn số nước"
        self.status.configure(text=f"Kết thúc: {result} ({reason})")
        self.write_log(f"\n=== Kết thúc: {result} ({reason}) ===\n", "move")
        self.save_pgn(result)

    def save_pgn(self, result):
        game = chess.pgn.Game.from_board(self.board)
        game.headers["Event"] = "Chess AI Arena"
        game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
        game.headers["White"] = self.players[chess.WHITE].name
        game.headers["Black"] = self.players[chess.BLACK].name
        game.headers["Result"] = result
        os.makedirs("games", exist_ok=True)
        path = os.path.join("games", datetime.datetime.now().strftime("game_%Y%m%d_%H%M%S.pgn"))
        with open(path, "w", encoding="utf-8") as f:
            print(game, file=f)
        self.write_log(f"Đã lưu PGN: {path}\n", "note")

    def on_close(self):
        for p in self.players.values():
            p.close()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description="Chess AI Arena")
    ap.add_argument("--white", default="claude:claude-sonnet-5-5")
    ap.add_argument("--black", default="claude:claude-haiku-5-5")
    ap.add_argument("--delay", type=float, default=1.0, help="giây nghỉ giữa các nước")
    ap.add_argument("--max-plies", type=int, default=200, help="giới hạn nửa-nước (ply)")
    args = ap.parse_args()

    white, black = make_player(args.white), make_player(args.black)
    root = tk.Tk()
    App(root, white, black, args.delay, args.max_plies)
    root.mainloop()


if __name__ == "__main__":
    main()
