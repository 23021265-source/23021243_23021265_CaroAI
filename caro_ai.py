"""
Bài tập giữa kỳ – Cờ Caro AI (9x9, 4 quân thắng)
Thuật toán: Minimax và Alpha-Beta Pruning (có thể chọn)
Yêu cầu: Python 3.8+, numpy, tkinter (có sẵn trong stdlib)
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import time
import threading

# ─────────────────────── HẰNG SỐ ───────────────────────
BOARD_SIZE = 9
EMPTY  = 0
PLAYER = 1   # X – người chơi
AI     = 2   # O – máy
WIN_LENGTH = 4
DEFAULT_DEPTH = 3

SCORE_WIN   =  1_000_000
SCORE_LOSE  = -1_000_000
SCORE_3_AI  =  1_000
SCORE_2_AI  =  100
SCORE_1_AI  =  10
SCORE_3_PLR = -2_000   # âm lớn → ưu tiên chặn
SCORE_2_PLR = -200
SCORE_1_PLR = -20

# ─────────────────────── LOGIC CHUNG ───────────────────────

def check_win(board, r, c, p):
    """Kiểm tra p có 4 quân liên tiếp sau khi đánh vào (r,c) không."""
    directions = [(0,1),(1,0),(1,1),(1,-1)]
    for dr, dc in directions:
        count = 1
        for sign in (1, -1):
            for i in range(1, WIN_LENGTH):
                nr, nc = r + sign*dr*i, c + sign*dc*i
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == p:
                    count += 1
                else:
                    break
        if count >= WIN_LENGTH:
            return True
    return False

def is_full(board):
    return not np.any(board == EMPTY)

def get_valid_moves(board):
    """
    Sinh các ô trống cách quân đã đánh ≤ 2 ô.
    Sắp xếp ưu tiên ô gần trung tâm (move ordering).
    """
    occupied = list(zip(*np.where(board != EMPTY)))
    if not occupied:
        center = BOARD_SIZE // 2
        return [(center, center)]
    moves = set()
    for (r, c) in occupied:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == EMPTY:
                    moves.add((nr, nc))
    center = BOARD_SIZE // 2
    return sorted(moves, key=lambda m: abs(m[0]-center) + abs(m[1]-center))

def score_window(window):
    """
    Chấm điểm 1 cửa sổ WIN_LENGTH ô.
    Theo gợi ý hàm đánh giá trong đề bài (mục 3a).
    """
    ai_cnt  = window.count(AI)
    plr_cnt = window.count(PLAYER)
    if ai_cnt > 0 and plr_cnt > 0:
        return 0
    if ai_cnt  == 4: return SCORE_WIN
    if plr_cnt == 4: return SCORE_LOSE
    if ai_cnt  == 3: return SCORE_3_AI
    if plr_cnt == 3: return SCORE_3_PLR
    if ai_cnt  == 2: return SCORE_2_AI
    if plr_cnt == 2: return SCORE_2_PLR
    if ai_cnt  == 1: return SCORE_1_AI
    if plr_cnt == 1: return SCORE_1_PLR
    return 0

def evaluate_board(board):
    """
    Hàm đánh giá heuristic – duyệt 4 hướng,
    mỗi chuỗi chỉ đếm 1 lần (tránh double-count).
    """
    score = 0
    directions = [(0,1),(1,0),(1,1),(1,-1)]
    for dr, dc in directions:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                pr, pc = r - dr, c - dc
                if (0 <= pr < BOARD_SIZE and 0 <= pc < BOARD_SIZE
                        and board[pr][pc] == board[r][c] != EMPTY):
                    continue
                seg, nr, nc = [], r, c
                while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    seg.append(int(board[nr][nc]))
                    nr += dr; nc += dc
                for i in range(len(seg) - WIN_LENGTH + 1):
                    score += score_window(seg[i:i+WIN_LENGTH])
    return score


# ─────────────────────── ENGINE AI ───────────────────────

class AIEngine:
    """
    Tách riêng engine AI để tái sử dụng trong benchmark.
    mode = 'minimax' | 'alphabeta'
    """
    def __init__(self, mode='alphabeta'):
        self.mode = mode
        self.nodes_visited = 0

    # ── Minimax thuần ────────────────────────────────────
    def minimax(self, board, depth, maximizing):
        self.nodes_visited += 1
        moves = get_valid_moves(board)
        if not moves or depth == 0:
            return None, evaluate_board(board)
        best_move = moves[0]
        if maximizing:
            best_val = -float('inf')
            for r, c in moves:
                board[r][c] = AI
                if check_win(board, r, c, AI):
                    board[r][c] = EMPTY
                    return (r,c), SCORE_WIN + depth*10
                _, val = self.minimax(board, depth-1, False)
                board[r][c] = EMPTY
                if val > best_val:
                    best_val = val; best_move = (r,c)
            return best_move, best_val
        else:
            best_val = float('inf')
            for r, c in moves:
                board[r][c] = PLAYER
                if check_win(board, r, c, PLAYER):
                    board[r][c] = EMPTY
                    return (r,c), SCORE_LOSE - depth*10
                _, val = self.minimax(board, depth-1, True)
                board[r][c] = EMPTY
                if val < best_val:
                    best_val = val; best_move = (r,c)
            return best_move, best_val

    # ── Alpha-Beta pruning ────────────────────────────────
    def alphabeta(self, board, depth, alpha, beta, maximizing):
        self.nodes_visited += 1
        moves = get_valid_moves(board)
        if not moves or depth == 0:
            return None, evaluate_board(board)
        best_move = moves[0]
        if maximizing:
            best_val = -float('inf')
            for r, c in moves:
                board[r][c] = AI
                if check_win(board, r, c, AI):
                    board[r][c] = EMPTY
                    return (r,c), SCORE_WIN + depth*10
                _, val = self.alphabeta(board, depth-1, alpha, beta, False)
                board[r][c] = EMPTY
                if val > best_val:
                    best_val = val; best_move = (r,c)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return best_move, best_val
        else:
            best_val = float('inf')
            for r, c in moves:
                board[r][c] = PLAYER
                if check_win(board, r, c, PLAYER):
                    board[r][c] = EMPTY
                    return (r,c), SCORE_LOSE - depth*10
                _, val = self.alphabeta(board, depth-1, alpha, beta, True)
                board[r][c] = EMPTY
                if val < best_val:
                    best_val = val; best_move = (r,c)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return best_move, best_val

    def best_move(self, board, depth):
        """Trả về (move, score, nodes_visited, elapsed_time)."""
        self.nodes_visited = 0
        t0 = time.time()
        b  = board.copy()
        if self.mode == 'minimax':
            move, score = self.minimax(b, depth, True)
        else:
            move, score = self.alphabeta(b, depth, -float('inf'), float('inf'), True)
        return move, score, self.nodes_visited, time.time() - t0


# ─────────────────────── GIAO DIỆN CHÍNH ───────────────────────

class CaroApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cờ Caro AI – Minimax & Alpha-Beta (9×9)")
        self.root.resizable(False, False)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.buttons = [[None]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.game_over = False
        self.engine = AIEngine(mode='alphabeta')
        self._build_ui()

    # ── Xây UI ──────────────────────────────────────────────
    def _build_ui(self):
        # Thanh điều khiển
        ctrl = tk.Frame(self.root, pady=5)
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8)

        tk.Label(ctrl, text="Thuật toán AI:").pack(side="left")
        self.algo_var = tk.StringVar(value="alphabeta")
        tk.Radiobutton(ctrl, text="Alpha-Beta", variable=self.algo_var,
                       value="alphabeta", command=self._on_algo).pack(side="left")
        tk.Radiobutton(ctrl, text="Minimax",    variable=self.algo_var,
                       value="minimax",    command=self._on_algo).pack(side="left")

        tk.Label(ctrl, text="   Độ sâu:").pack(side="left")
        self.depth_var = tk.IntVar(value=DEFAULT_DEPTH)
        tk.Spinbox(ctrl, from_=1, to=6, width=3,
                   textvariable=self.depth_var).pack(side="left", padx=2)

        tk.Button(ctrl, text="  Chơi lại  ", bg="#27ae60", fg="white",
                  command=self.reset_game).pack(side="left", padx=10)
        tk.Button(ctrl, text="  Benchmark  ", bg="#8e44ad", fg="white",
                  command=self._open_benchmark).pack(side="left")

        # Bàn cờ
        board_frame = tk.Frame(self.root, bg="#95a5a6", padx=2, pady=2)
        board_frame.grid(row=1, column=0, padx=8, pady=4)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                btn = tk.Button(board_frame, text="", font=('Arial', 11, 'bold'),
                                width=4, height=2, bg="#ecf0f1",
                                command=lambda row=r, col=c: self.player_move(row, col))
                btn.grid(row=r, column=c, padx=1, pady=1)
                self.buttons[r][c] = btn

        # Bảng thống kê
        info = tk.LabelFrame(self.root, text="Thông tin nước đi AI", padx=8, pady=8)
        info.grid(row=1, column=1, sticky="n", padx=8, pady=4)

        rows = [
            ("Thuật toán",          "algo"),
            ("Nước đi (hàng, cột)", "move"),
            ("Giá trị đánh giá",    "score"),
            ("Số trạng thái đã xét","nodes"),
            ("Thời gian (giây)",    "time"),
            ("Độ sâu tìm kiếm",     "depth"),
        ]
        self.stat_vars = {}
        for i, (label, key) in enumerate(rows):
            tk.Label(info, text=label+":", anchor="w").grid(
                row=i, column=0, sticky="w", pady=3, padx=2)
            var = tk.StringVar(value="–")
            tk.Label(info, textvariable=var, font=('Courier', 10, 'bold'),
                     fg="#1a252f", width=18, anchor="w").grid(
                row=i, column=1, sticky="w")
            self.stat_vars[key] = var

        # Log nước đi
        log_frame = tk.LabelFrame(self.root, text="Log nước đi", padx=4, pady=4)
        log_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        self.log = scrolledtext.ScrolledText(
            log_frame, height=6, width=78, font=('Courier', 9), state="disabled")
        self.log.pack(fill="both")

    # ── Tiện ích ────────────────────────────────────────────
    def _on_algo(self):
        self.engine.mode = self.algo_var.get()

    def _log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _update_stats(self, algo, move, score, nodes, elapsed, depth):
        self.stat_vars["algo"].set(algo)
        self.stat_vars["move"].set(f"hàng {move[0]+1}, cột {move[1]+1}")
        self.stat_vars["score"].set(str(score))
        self.stat_vars["nodes"].set(f"{nodes:,}")
        self.stat_vars["time"].set(f"{elapsed:.4f}")
        self.stat_vars["depth"].set(str(depth))

    def _set_board_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == EMPTY:
                    self.buttons[r][c].config(state=state)

    # ── Luồng chơi ──────────────────────────────────────────
    def player_move(self, r, c):
        if self.game_over or self.board[r][c] != EMPTY:
            return
        self._place(r, c, PLAYER)
        if check_win(self.board, r, c, PLAYER):
            self._end("Bạn thắng! 🎉"); return
        if is_full(self.board):
            self._end("Hòa!"); return
        self._set_board_enabled(False)
        threading.Thread(target=self._ai_worker, daemon=True).start()

    def _ai_worker(self):
        """Thread phụ – chỉ tính toán, không cập nhật UI."""
        depth = self.depth_var.get()
        self.engine.mode = self.algo_var.get()
        result = self.engine.best_move(self.board, depth)
        self.root.after(0, self._ai_done, *result, depth)

    def _ai_done(self, move, score, nodes, elapsed, depth):
        """Main thread – cập nhật UI sau khi AI tính xong."""
        if not move or self.game_over:
            return
        r, c = move
        name = "Alpha-Beta" if self.engine.mode == "alphabeta" else "Minimax"
        self._update_stats(name, move, score, nodes, elapsed, depth)
        line = (f"[{name}] AI→({r+1},{c+1}) | score={score} "
                f"| nodes={nodes:,} | time={elapsed:.4f}s | depth={depth}")
        self._log(line)
        print(line)

        self._place(r, c, AI)
        if check_win(self.board, r, c, AI):
            self._end("Máy thắng!"); return
        if is_full(self.board):
            self._end("Hòa!"); return
        self._set_board_enabled(True)

    def _place(self, r, c, player):
        self.board[r][c] = player
        color = "#e74c3c" if player == PLAYER else "#2980b9"
        self.buttons[r][c].config(
            text=("X" if player == PLAYER else "O"),
            fg="white", bg=color, state="disabled")

    def _end(self, msg):
        self.game_over = True
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == EMPTY:
                    self.buttons[r][c].config(state="disabled")
        self._log(f"=== {msg} ===")
        messagebox.showinfo("Kết thúc", msg)

    def reset_game(self):
        self.board.fill(EMPTY)
        self.game_over = False
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.buttons[r][c].config(text="", bg="#ecf0f1", state="normal")

    def _open_benchmark(self):
        BenchmarkWindow(self.root)


# ─────────────────────── 5 TRẠNG THÁI KIỂM THỬ ───────────────────────
# Theo yêu cầu Level 3: ít nhất 5 trạng thái khác nhau
TEST_STATES = {
    "1. Đầu ván (trống)":
        np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int),

    "2. Giữa ván":
        np.array([
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,1,2,0,0,0,0],
            [0,0,1,2,1,0,0,0,0],
            [0,0,2,1,2,1,0,0,0],
            [0,0,0,0,1,2,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
        ], dtype=int),

    "3. Máy sắp thắng":
        np.array([
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,1,1,0,0,0,0,0],
            [0,0,0,2,2,2,0,0,0],
            [0,0,0,1,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
        ], dtype=int),

    "4. Người sắp thắng (máy chặn)":
        np.array([
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,2,0,0,0,0,0,0],
            [0,0,0,2,0,0,0,0,0],
            [0,0,1,1,1,0,0,0,0],
            [0,0,0,0,0,2,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
        ], dtype=int),

    "5. Hai bên cùng tấn công":
        np.array([
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,0,0,0,0],
            [0,0,0,2,0,0,0,0,0],
            [0,0,0,2,2,0,0,0,0],
            [0,0,1,0,2,1,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
        ], dtype=int),
}


# ─────────────────────── BENCHMARK WINDOW ───────────────────────

class BenchmarkWindow:
    """
    Chạy Minimax và Alpha-Beta trên cùng trạng thái + độ sâu,
    ghi nhận: nước đi, giá trị, số trạng thái, thời gian (Level 2, 3).
    """
    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Benchmark – Minimax vs Alpha-Beta")
        self.win.geometry("900x480")
        self._build()

    def _build(self):
        top = tk.Frame(self.win, pady=6)
        top.pack(fill="x", padx=8)

        tk.Label(top, text="Độ sâu:").pack(side="left")
        self.depth_var = tk.IntVar(value=3)
        tk.Spinbox(top, from_=1, to=5, width=4,
                   textvariable=self.depth_var).pack(side="left", padx=4)

        tk.Label(top, text="  Thử thêm độ sâu:").pack(side="left")
        self.multi_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Chạy depth 1,2,3 cùng lúc",
                       variable=self.multi_var).pack(side="left")

        tk.Button(top, text="▶  Chạy benchmark", bg="#2980b9", fg="white",
                  command=self._run).pack(side="left", padx=10)
        self.status = tk.StringVar(value="Nhấn ▶ để bắt đầu.")
        tk.Label(top, textvariable=self.status, fg="#555").pack(side="left")

        cols = ("Trạng thái", "Depth",
                "MM nước đi", "MM nodes",  "MM time(s)",
                "AB nước đi", "AB nodes",  "AB time(s)",
                "Cùng nước đi?", "Giảm nodes(%)")
        self.tree = ttk.Treeview(self.win, columns=cols, show="headings", height=14)
        for col in cols:
            w = 130 if "Trạng thái" in col else 85
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")

        sb = ttk.Scrollbar(self.win, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8,0), pady=4)
        sb.pack(side="left", fill="y", pady=4, padx=(0,8))

    def _run(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.status.set("Đang chạy…")
        self.win.update()

        base_depth = self.depth_var.get()
        depths = list(range(1, base_depth+1)) if self.multi_var.get() else [base_depth]

        mm = AIEngine(mode='minimax')
        ab = AIEngine(mode='alphabeta')

        for depth in depths:
            for name, state in TEST_STATES.items():
                mm_move, mm_score, mm_nodes, mm_t = mm.best_move(state.copy(), depth)
                ab_move, ab_score, ab_nodes, ab_t = ab.best_move(state.copy(), depth)

                same = "✓" if mm_move == ab_move else "✗"
                pct  = f"{(1 - ab_nodes/max(mm_nodes,1))*100:.1f}%" if mm_nodes else "–"
                mm_s = f"({mm_move[0]+1},{mm_move[1]+1})" if mm_move else "–"
                ab_s = f"({ab_move[0]+1},{ab_move[1]+1})" if ab_move else "–"

                self.tree.insert("", "end", values=(
                    name[:20], depth,
                    mm_s, f"{mm_nodes:,}", f"{mm_t:.3f}",
                    ab_s, f"{ab_nodes:,}", f"{ab_t:.3f}",
                    same, pct
                ))

        self.status.set(f"Xong! {len(TEST_STATES)*len(depths)} bài kiểm thử.")


# ─────────────────────── MAIN ────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    CaroApp(root)
    root.mainloop()
