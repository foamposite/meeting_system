# --- 1. 顶部只保留 Python 自带的轻量级标准库 (0.1秒极速加载) ---
import os
import sys
import pickle
import datetime
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

class NullWriter:
    def write(self, text): pass
    def flush(self): pass

# 必须加这几行，否则没有控制台时打印日志会崩
if sys.stdout is None: sys.stdout = NullWriter()
if sys.stderr is None: sys.stderr = NullWriter()

# 🌟 注意：移除了 cv2, numpy, insightface, PIL 等重量级库的顶部导入！🌟

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

base_path = get_base_path()

def load_config():
    config_path = os.path.join(base_path, 'config.json')
    default_config = {
        "camera_id": 0, "resolution": [1280, 720], "threshold": 0.5,
        "db_folder": "database_images", "db_file": "face_db.pkl", "log_file": "checkin_log.csv"
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                default_config.update(json.load(f))
        except: pass
    return default_config

CONFIG = load_config()

# --- 2. 核心算法类 ---
class FaceSystem:
    def __init__(self):
        self.db_folder = os.path.join(base_path, CONFIG['db_folder'])
        self.db_file = os.path.join(base_path, CONFIG['db_file'])
        model_root = os.path.join(base_path, 'models')
        
        # 此时 insightface 已经在后台线程中被全局导入了，可以直接用
        self.app = FaceAnalysis(name='buffalo_l', root=model_root, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.known_embeddings = None
        self.known_names = []
        self.load_database()

    def load_database(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_names = data['names']
                    self.known_embeddings = data['embeddings']
                return True
            except: pass
        return False

    def register_faces(self):
        if not os.path.exists(self.db_folder):
            os.makedirs(self.db_folder)
            return 0, "文件夹不存在，已自动创建。请放入照片。"
        
        embeddings_list, names_list = [], []
        count = 0
        for filename in os.listdir(self.db_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(self.db_folder, filename)
                try:
                    img_data = np.fromfile(img_path, dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if img is None: continue
                    faces = self.app.get(img)
                    if len(faces) == 1:
                        embeddings_list.append(faces[0].embedding)
                        names_list.append(name)
                        count += 1
                except: continue
        
        if embeddings_list:
            self.known_embeddings = normalize(np.array(embeddings_list))
            self.known_names = names_list
            with open(self.db_file, 'wb') as f:
                pickle.dump({'names': names_list, 'embeddings': self.known_embeddings}, f)
            return count, "底库更新成功"
        return 0, "未检测到有效单人照片"

    def recognize(self, frame):
        faces = self.app.get(frame)
        results = []
        if len(faces) > 0 and self.known_embeddings is not None:
            current_embs = normalize(np.array([face.embedding for face in faces]))
            sim_matrix = np.dot(current_embs, self.known_embeddings.T)
            for i, face in enumerate(faces):
                max_idx = np.argmax(sim_matrix[i])
                max_score = sim_matrix[i][max_idx]
                name = self.known_names[max_idx] if max_score > CONFIG['threshold'] else "Unknown"
                results.append((face.bbox.astype(int), name, max_score))
        return results

    def save_log(self, names):
        if not names: return
        log_path = os.path.join(base_path, CONFIG['log_file'])
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, 'a', encoding='utf-8') as f:
            for name in set(names):
                if name != "Unknown": f.write(f"{timestamp},{name},Present\n")

# --- 3. UI 界面类 ---
# --- 3. UI 界面类 (含摄像头自动搜索与切换) ---
class AttendanceApp:
    def __init__(self, root, system=None):
        self.root = root
        self.system = system 
        self.root.title("会议签到系统")
        self.root.geometry("1000x700")
        
        self.is_camera_on = False
        self.cap = None
        self.is_paused = False
        self.available_cameras = [] # 存储可用摄像头索引
        
        self.setup_ui()
        
        # 启动时自动搜索摄像头
        self.refresh_camera_list()
        
        if self.system is None:
            self.status_var.set("界面已启动，正在后台加载核心库，请稍候...")

    def setup_ui(self):
        control_frame = ttk.LabelFrame(self.root, text="操作控制台", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # --- 模式一：实时监控区域 (改造) ---
        ttk.Label(control_frame, text="模式一：视频签到").pack(side=tk.LEFT, padx=(5, 5))
        
        # 1. 新增摄像头下拉选择框
        self.camera_combo = ttk.Combobox(control_frame, width=15, state="readonly")
        self.camera_combo.pack(side=tk.LEFT, padx=5)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_change) # 绑定切换事件
        
        # 2. 刷新按钮 (防止插拔摄像头后找不到)
        ttk.Button(control_frame, text="↻", width=3, command=self.refresh_camera_list).pack(side=tk.LEFT, padx=2)
        
        # 3. 打开/关闭按钮
        self.btn_camera = ttk.Button(control_frame, text="打开摄像头", command=self.toggle_camera)
        self.btn_camera.pack(side=tk.LEFT, padx=5)
        
        self.btn_snap = ttk.Button(control_frame, text="拍摄并签到", command=self.snap_and_check_in, state=tk.DISABLED)
        self.btn_snap.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=20)
        
        # --- 模式二：照片上传 ---
        ttk.Label(control_frame, text="模式二：照片签到").pack(side=tk.LEFT, padx=(10, 10))
        ttk.Button(control_frame, text="上传合影", command=self.upload_and_check_in).pack(side=tk.LEFT, padx=5)
        
        # --- 右侧功能 ---
        ttk.Button(control_frame, text="更新底库", command=self.update_db).pack(side=tk.RIGHT, padx=5)
        
        # 状态栏
        self.status_var = tk.StringVar(value="启动中...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.video_frame = ttk.Label(self.root, text="实时摄像头画面或上传的照片将显示在此处", relief=tk.RIDGE, anchor=tk.CENTER)
        self.video_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    # --- 新增：自动搜索可用摄像头 ---
    def refresh_camera_list(self):
        self.available_cameras = []
        combo_list = []
        self.status_var.set("正在扫描可用摄像头设备...")
        self.root.update()
        
        # 扫描索引 0 到 3 (通常电脑不会超过4个摄像头)
        for i in range(4):
            # Windows下使用 CAP_DSHOW 会更快，但为了兼容性先用默认
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) 
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.available_cameras.append(i)
                    combo_list.append(f"摄像头 {i}")
                cap.release()
        
        if not self.available_cameras:
            combo_list = ["未检测到设备"]
            self.btn_camera.config(state=tk.DISABLED)
        else:
            self.btn_camera.config(state=tk.NORMAL)
            
        self.camera_combo['values'] = combo_list
        if self.available_cameras:
            self.camera_combo.current(0) # 默认选中第一个
            
        self.status_var.set(f"扫描完成，发现 {len(self.available_cameras)} 个设备。")

    # --- 新增：切换摄像头时的逻辑 ---
    def on_camera_change(self, event):
        # 如果摄像头正在运行，直接切换
        if self.is_camera_on:
            self.stop_camera()
            self.root.after(500, self.start_camera) # 延迟一点点再开启，防止资源冲突

    def toggle_camera(self):
        if self.is_camera_on:
            self.stop_camera()
            self.btn_camera.config(text="打开摄像头")
        else:
            self.start_camera()
            if self.is_camera_on: # 如果成功启动
                self.btn_camera.config(text="关闭摄像头")

    def check_system_ready(self):
        if self.system is None:
            messagebox.showinfo("提示", "底层组件仍在加载中，请等待底部状态栏显示“就绪”后再操作...")
            return False
        return True

    def render_image_to_ui(self, frame_cv):
        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((960, 540), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk 
        self.video_frame.configure(image=imgtk)

    def start_camera(self):
        if not self.check_system_ready(): return
        if self.is_camera_on: return
        
        # 获取当前下拉框选中的索引
        try:
            selected_idx = self.camera_combo.current()
            if selected_idx < 0 or selected_idx >= len(self.available_cameras):
                messagebox.showerror("错误", "请先选择一个有效的摄像头")
                return
            camera_id = self.available_cameras[selected_idx]
        except:
            camera_id = 0

        try:
            # 优先使用 DirectShow (Windows优化)，启动更快
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # 如果失败，回退到默认驱动
                self.cap = cv2.VideoCapture(camera_id)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['resolution'][0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['resolution'][1])
            
            self.is_camera_on = True
            self.is_paused = False 
            self.btn_snap.config(state=tk.NORMAL)
            self.status_var.set(f"正在使用摄像头 {camera_id} 运行中。")
            self.update_frame()
        except Exception as e:
            messagebox.showerror("错误", f"无法打开摄像头 {camera_id}: {e}")
            self.is_camera_on = False

    def update_frame(self):
        if self.is_camera_on and self.cap.isOpened() and not self.is_paused:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.render_image_to_ui(frame)
            else:
                self.status_var.set("警告：无法从摄像头获取画面")
        
        if self.is_camera_on:
            self.root.after(30, self.update_frame)

    def stop_camera(self):
        self.is_camera_on = False
        if self.cap: self.cap.release()
        self.video_frame.configure(image='')
        self.btn_snap.config(state=tk.DISABLED)
        self.status_var.set("摄像头已关闭。")

    def process_image(self, frame):
        self.is_paused = True 
        self.status_var.set("正在使用 AI 识别图像... 请稍候。")
        self.root.update()
        
        results = self.system.recognize(frame)
        present_names = set()
        frame_to_save, frame_to_show = frame.copy(), frame.copy()
        
        for (bbox, name, score) in results:
            x1, y1, x2, y2 = bbox
            color_cv = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            color_pil = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
            
            cv2.rectangle(frame_to_save, (x1, y1), (x2, y2), color_cv, 2)
            frame_to_save = self.draw_chinese_text(frame_to_save, f"{name} ({score:.2f})", (x1, y1-25), color_pil, 20)
            
            cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), color_cv, 2)
            display_name = name if name != "Unknown" else "Unknown"
            frame_to_show = self.draw_chinese_text(frame_to_show, display_name, (x1, y1-25), color_pil, 20)
            
            if name != "Unknown": present_names.add(name)

        self.render_image_to_ui(frame_to_show)
        self.system.save_log(list(present_names))
        self.save_evidence_photo(frame_to_save)
        self.status_var.set(f"签到处理完毕。成功识别 {len(present_names)} 人。")
        self.show_result_window(present_names, len(results))

    def snap_and_check_in(self):
        if not self.check_system_ready(): return
        if not self.is_camera_on or not hasattr(self, 'current_frame'): return
        self.process_image(self.current_frame.copy())

    def upload_and_check_in(self):
        if not self.check_system_ready(): return
        file_path = filedialog.askopenfilename(title="选择合影", filetypes=[("图片", "*.jpg *.png *.bmp")])
        if file_path:
            self.status_var.set(f"正在加载图片...")
            self.root.update()
            try:
                frame = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    # 如果摄像头开着，先暂时关闭（暂停），避免画面冲突
                    was_camera_on = self.is_camera_on
                    if was_camera_on:
                        self.is_paused = True
                    
                    self.current_frame = frame 
                    self.process_image(frame)
            except Exception as e:
                messagebox.showerror("错误", f"无法加载文件: {e}")

    def show_result_window(self, present_names, total_detected):
        result_win = tk.Toplevel(self.root)
        result_win.title("会议签到明细")
        result_win.geometry("400x500")
        
        def on_close():
            self.is_paused = False # 恢复实时画面
            if self.is_camera_on: self.status_var.set("已恢复实时监控。")
            result_win.destroy()
            
        result_win.protocol("WM_DELETE_WINDOW", on_close)
        
        ttk.Label(result_win, text=f"人脸总数: {total_detected}\n成功签到: {len(present_names)}", font=("微软雅黑", 12, "bold"), padding=15, justify=tk.CENTER).pack()
        
        tree = ttk.Treeview(result_win, columns=("name", "status"), show="headings", selectmode="none")
        tree.heading("name", text="姓名"); tree.column("name", width=200, anchor=tk.CENTER)
        tree.heading("status", text="状态"); tree.column("status", width=150, anchor=tk.CENTER)
        tree.pack(expand=True, fill=tk.BOTH, padx=20, pady=5)
        
        if not present_names:
             tree.insert("", tk.END, values=("未能识别出库中人员", "❌ 未签到"), tags=('unknown',))
        else:
            for name in sorted(list(present_names)):
                tree.insert("", tk.END, values=(name, "✅ 已打卡"), tags=('present',))
        
        tree.tag_configure('present', foreground='green'); tree.tag_configure('unknown', foreground='red')
        ttk.Button(result_win, text="关闭窗口并继续", command=on_close).pack(pady=15)

    def save_evidence_photo(self, frame):
        try:
            photo_dir = os.path.join(base_path, "attendance_photos")
            if not os.path.exists(photo_dir): os.makedirs(photo_dir)
            filepath = os.path.join(photo_dir, f"打卡证据_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
            cv2.imencode('.jpg', frame)[1].tofile(filepath)
        except: pass

    def draw_chinese_text(self, img_cv, text, position, color=(0, 255, 0), text_size=20):
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try: font = ImageFont.truetype("msyh.ttc", text_size)
        except: font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def update_db(self):
        if not self.check_system_ready(): return
        self.status_var.set("正在更新底库...")
        self.root.update()
        count, msg = self.system.register_faces()
        messagebox.showinfo("底库更新", f"{msg}\n库中总人数: {count}")
        self.status_var.set("底库更新完毕。")

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

# --- 4. 程序入口 (真·异步启动逻辑) ---
if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass

    root = tk.Tk()
    app = AttendanceApp(root, system=None) 
    # 强制立刻绘制 UI 界面
    root.update()
    
    def load_ai_thread():
        try:
            app.status_var.set("正在导入视觉算法库 (这可能需要几秒钟)...")
            
            # --- ✅ 修正点 2：这里只导入原本最慢的库 ---
            # cv2, np, PIL 已经在上面导入了，这里不需要再 import
            
            # 🌟 核心魔法：只在这里导入 InsightFace 和 sklearn 🌟
            global insightface, FaceAnalysis, normalize
            import insightface
            from insightface.app import FaceAnalysis
            from sklearn.preprocessing import normalize # sklearn 也比较慢，建议留在这里
            
            app.status_var.set("正在初始化 AI 模型，即将完成...")
            face_sys = FaceSystem() # 初始化模型
            
            root.after(0, on_ai_loaded, face_sys) 
        except Exception as e:
            root.after(0, on_ai_error, str(e))

    def on_ai_loaded(face_sys):
        app.system = face_sys
        app.status_var.set("AI 模型加载完毕，系统就绪。请选择模式打卡。")

    def on_ai_error(err_msg):
        app.status_var.set("AI 模型加载失败！")
        messagebox.showerror("致命错误", f"模型加载失败，请检查 models 文件夹。\n{err_msg}")

    # 启动后台线程
    threading.Thread(target=load_ai_thread, daemon=True).start()

    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()