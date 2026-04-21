import cv2
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from pathlib import Path

# =========================================================
# THEME & CONFIG (SOFT PALETTE)
# =========================================================
WHITE = "#FFFFFF"
PALE_GREEN = "#F1F8E9"  
HOVER_GREEN = "#DCEDC8"
TEXT_MAIN = "#455A64"
ACCENT_HEX = "#81C784" 

# Soft Border Colors
SHADOW_OUTER = "#DCDDE1"
SHADOW_INNER = "#F5F6FA"

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "smart_config.json"

def hex_to_bgr(hex_str):
    hex_str = hex_str.lstrip('#')
    lv = len(hex_str)
    rgb = tuple(int(hex_str[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0]) 

C_ACCENT = hex_to_bgr(ACCENT_HEX)
C_ORANGE, C_BLUE = (0, 165, 255), (255, 0, 0)
C_AMBER, C_CYAN = (0, 191, 255), (255, 255, 0)

# =========================================================
# GLOBAL STATE
# =========================================================
lines, points_zebra, points_buffer = [], [], []
selected_point = None
current_zone_mode = "ZEBRA"
is_frozen = False

def update_json(key, data):
    full_config = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f: full_config = json.load(f)
        except: pass
    full_config[key] = data
    with open(CONFIG_PATH, "w") as f: json.dump(full_config, f, indent=4)

# =========================================================
# SOFT-EDGE WINDOW COMPONENT
# =========================================================
class AestheticWindow(tk.Toplevel):
    def __init__(self, master, title_text, subtitle):
        super().__init__(master)
        self.width, self.height = 380, 480
        self.withdraw()
        
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f'{self.width}x{self.height}+{(sw-self.width)//2}+{(sh-self.height)//2}')
        self.overrideredirect(True) 
        
        # Triple-frame layering to soften the perimeter
        self.shadow_frame = tk.Frame(self, bg=SHADOW_OUTER, bd=1)
        self.shadow_frame.pack(fill="both", expand=True)
        
        self.glow_frame = tk.Frame(self.shadow_frame, bg=SHADOW_INNER, bd=2)
        self.glow_frame.pack(fill="both", expand=True)
        
        self.main_frame = tk.Frame(self.glow_frame, bg=WHITE)
        self.main_frame.pack(fill="both", expand=True)

        self.main_frame.bind("<ButtonPress-1>", self.start_move)
        self.main_frame.bind("<B1-Motion>", self.do_move)

        # High-visibility "✕" button
        self.close_btn = tk.Label(self.main_frame, text="✕", bg=WHITE, fg="#95A5A6", 
                                 font=("Arial", 14, "bold"), cursor="hand2")
        self.close_btn.place(x=340, y=15)
        self.close_btn.bind("<Button-1>", lambda e: self.destroy())

        tk.Label(self.main_frame, text=title_text, font=("Verdana", 16, "bold"), bg=WHITE, fg=ACCENT_HEX).pack(pady=(60, 5))
        tk.Label(self.main_frame, text=subtitle, font=("Verdana", 7, "bold"), bg=WHITE, fg="#B0BEC5").pack(pady=(0, 40))

        self.deiconify()
        self.attributes("-alpha", 0.0)
        self.fade_in()

    def fade_in(self):
        alpha = self.attributes("-alpha")
        if alpha < 1.0:
            self.attributes("-alpha", alpha + 0.1)
            self.after(15, self.fade_in)

    def start_move(self, event): self.x, self.y = event.x, event.y
    def do_move(self, event):
        self.geometry(f"+{self.winfo_x() + event.x - self.x}+{self.winfo_y() + event.y - self.y}")

    def create_button(self, text, command, color=PALE_GREEN, t_color=TEXT_MAIN):
        # The 'Outer' Frame acts as a rounded padding area
        btn_padding = tk.Frame(self.main_frame, bg=color, bd=0)
        btn_padding.pack(pady=10, padx=20)
        
        # Inner button with zero relief to look like a flat soft card
        btn = tk.Button(btn_padding, text=text, command=command, width=24, height=1,
                        bg=color, fg=t_color, relief="flat", font=("Verdana", 9, "bold"),
                        activebackground=HOVER_GREEN, cursor="hand2", bd=0, 
                        padx=15, pady=8) # Extra internal padding for "pill" feel
        btn.pack()
        
        def on_ent(e): btn.config(bg=HOVER_GREEN); btn_padding.config(bg=HOVER_GREEN)
        def on_lev(e): btn.config(bg=color); btn_padding.config(bg=color)
        btn.bind("<Enter>", on_ent)
        btn.bind("<Leave>", on_lev)
        return btn

# =========================================================
# HUD & OPENCV INTERACTION
# =========================================================
def draw_modern_hud(img, mode):
    h, w = img.shape[:2]
    # Fixed Sidebar Background
    overlay = img.copy()
    cv2.rectangle(overlay, (w - 320, 0), (w, 280), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    
    start_x = w - 290
    cv2.putText(img, f"{mode.upper()} SETUP", (start_x, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (start_x, 65), (w - 30, 65), C_ACCENT, 2, cv2.LINE_AA)
    
    ctrls = [("F", "Freeze Stream"), ("R", "Reset Drawing"), ("S", "Save & Return"), ("ESC", "Exit")]
    if mode == 'zone': ctrls.insert(0, ("N", "Toggle Area"))

    for i, (k, d) in enumerate(ctrls):
        y = 110 + (i * 35)
        # Bold green keys for high visibility
        cv2.putText(img, f"[{k}]", (start_x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (129, 199, 132), 1, cv2.LINE_AA)
        cv2.putText(img, d, (start_x + 60, y), cv2.FONT_HERSHEY_DUPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

def speed_mouse(event, x, y, flags, param):
    global lines, selected_point
    dist = lambda p1, p2: np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    if event == cv2.EVENT_LBUTTONDOWN:
        for s_idx, shape in enumerate(lines):
            for p_idx, pt in enumerate(shape):
                if dist((x, y), pt) < 15: selected_point = (s_idx, p_idx); return
        if len(lines) < 2:
            lines.append([(x, y), (x + 100, y)]); selected_point = (len(lines)-1, 1)
    elif event == cv2.EVENT_MOUSEMOVE and selected_point:
        lines[selected_point[0]][selected_point[1]] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP: selected_point = None

def zone_mouse(event, x, y, flags, param):
    global points_zebra, points_buffer, selected_point, current_zone_mode
    active = points_zebra if current_zone_mode == "ZEBRA" else points_buffer
    dist = lambda p1, p2: np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(active):
            if dist((x, y), pt) < 15: selected_point = i; return
        active.append((x, y)); selected_point = len(active) - 1
    elif event == cv2.EVENT_MOUSEMOVE and selected_point is not None:
        active[selected_point] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP: selected_point = None
    elif event == cv2.EVENT_RBUTTONDOWN and active: active.pop()

def run_calibration(cap, mode):
    global current_zone_mode, is_frozen, lines, points_zebra, points_buffer
    cv2.namedWindow("Config Tool", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Config Tool", speed_mouse if mode == 'line' else zone_mouse)
    
    while True:
        if not is_frozen:
            ret, frame = cap.read()
            if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            last_frame = frame.copy()
        
        disp = last_frame.copy()
        draw_modern_hud(disp, mode)

        if mode == 'line':
            for i, line in enumerate(lines):
                c = C_ORANGE if i == 0 else C_BLUE
                cv2.line(disp, line[0], line[1], c, 3, cv2.LINE_AA)
                for pt in line: cv2.circle(disp, pt, 8, (255,255,255), -1); cv2.circle(disp, pt, 8, c, 2)
        else:
            for p_list, color in [(points_zebra, C_AMBER), (points_buffer, C_CYAN)]:
                if len(p_list) > 1:
                    pts = np.array(p_list, np.int32)
                    cv2.polylines(disp, [pts], len(p_list)>2, color, 3, cv2.LINE_AA)
                    if len(p_list)>2:
                        ov = disp.copy(); cv2.fillPoly(ov, [pts], color)
                        cv2.addWeighted(ov, 0.2, disp, 0.8, 0, disp)
                for pt in p_list: cv2.circle(disp, pt, 5, color, -1)
            cv2.putText(disp, f"ACTIVE: {current_zone_mode}", (30, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Config Tool", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'): is_frozen = not is_frozen
        if key == ord('r'): 
            if mode == 'line': lines.clear()
            else: points_zebra.clear() if current_zone_mode == "ZEBRA" else points_buffer.clear()
        if key == ord('n') and mode == 'zone': current_zone_mode = "BUFFER" if current_zone_mode == "ZEBRA" else "ZEBRA"
        if key == ord('s'):
            if mode == 'line' and len(lines) == 2: update_json("speed_lines", {"line1": lines[0], "line2": lines[1]}); break
            elif mode == 'zone' and len(points_zebra) > 2: update_json("parking_zones", {"zebra_zone": points_zebra, "buffer_zone": points_buffer}); break
        if key == 27: break
    cv2.destroyAllWindows()

# =========================================================
# MAIN APP FLOW
# =========================================================
def main():
    root = tk.Tk(); root.withdraw()

    source_type = [None]
    def select_src(v): source_type[0] = v; src_ui.destroy()

    src_ui = AestheticWindow(root, "CONFIGURATION", "CHOOSE VIDEO SOURCE")
    src_ui.create_button("VIDEO FILE", lambda: select_src('file'))
    src_ui.create_button("LOCAL WEBCAM", lambda: select_src('webcam'))
    src_ui.create_button("IP CAMERA (RTSP)", lambda: select_src('ip'))
    root.wait_window(src_ui)

    if not source_type[0]: return

    if source_type[0] == 'file': path = filedialog.askopenfilename()
    elif source_type[0] == 'webcam': path = 0
    else: path = simpledialog.askstring("RTSP", "Enter URL:")
    
    if path is None: return
    cap = cv2.VideoCapture(path)

    while True:
        task = [None]
        hub_ui = AestheticWindow(root, "DETECTION", "SELECT CALIBRATION TASK")
        def select_task(v): task[0] = v; hub_ui.destroy()
        
        hub_ui.create_button("SPEED LINES", lambda: select_task('line'))
        hub_ui.create_button("PARKING ZONES", lambda: select_task('zone'))
        hub_ui.create_button("FINISH CONFIG", lambda: select_task('exit'), color="#FFEBEE", t_color="#C62828")
        root.wait_window(hub_ui)

        if task[0] == 'exit' or task[0] is None: break
        run_calibration(cap, task[0])

    cap.release(); root.destroy()

if __name__ == "__main__":
    main()
