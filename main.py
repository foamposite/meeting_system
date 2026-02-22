import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import sys
import pickle
import datetime
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from sklearn.preprocessing import normalize
import platform

# --- 1. Path & Config Utils ---
# Same as before, ensures paths work for both script and EXE
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
                user_config = json.load(f)
                default_config.update(user_config)
        except Exception as e:
            print(f"Config load failed: {e}, using defaults.")
    return default_config

CONFIG = load_config()

# --- 2. Backend Logic (Unchanged) ---
class FaceSystem:
    def __init__(self):
        self.db_folder = os.path.join(base_path, CONFIG['db_folder'])
        self.db_file = os.path.join(base_path, CONFIG['db_file'])
        model_root = os.path.join(base_path, 'models')
        
        self.app = FaceAnalysis(
            name='buffalo_l', 
            root=model_root,
            providers=['CPUExecutionProvider']
        )
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
            except:
                return False
        return False

    def register_faces(self):
        if not os.path.exists(self.db_folder):
            os.makedirs(self.db_folder)
            return 0, "Folder created. Please add images."

        embeddings_list = []
        names_list = []
        count = 0

        for filename in os.listdir(self.db_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(self.db_folder, filename)
                try:
                    img_data = np.fromfile(img_path, dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                except: continue
                
                if img is None: continue
                
                faces = self.app.get(img)
                if len(faces) == 1:
                    embedding = faces[0].embedding
                    embeddings_list.append(embedding)
                    names_list.append(name)
                    count += 1
        
        if embeddings_list:
            self.known_embeddings = normalize(np.array(embeddings_list))
            self.known_names = names_list
            with open(self.db_file, 'wb') as f:
                pickle.dump({'names': names_list, 'embeddings': self.known_embeddings}, f)
            return count, "Database updated successfully."
        else:
            return 0, "No valid single-face images found."

    def recognize(self, frame):
        faces = self.app.get(frame)
        results = []
        if len(faces) > 0 and self.known_embeddings is not None:
            current_embs = normalize(np.array([face.embedding for face in faces]))
            sim_matrix = np.dot(current_embs, self.known_embeddings.T)
            
            for i, face in enumerate(faces):
                max_score = np.max(sim_matrix[i])
                max_idx = np.argmax(sim_matrix[i])
                name = "Unknown"
                if max_score > CONFIG['threshold']:
                    name = self.known_names[max_idx]
                results.append((face.bbox.astype(int), name, max_score))
        return results

    def save_log(self, names):
        if not names: return
        log_path = os.path.join(base_path, CONFIG['log_file'])
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, 'a', encoding='utf-8') as f:
            for name in set(names):
                if name != "Unknown":
                    f.write(f"{timestamp},{name},Present\n")

# --- 3. UI 前端界面 (全中文专业版) ---
class AttendanceApp:
    def __init__(self, root, system):
        self.root = root
        self.system = system
        self.root.title("会议签到系统")
        self.root.geometry("1000x700")
        
        self.is_camera_on = False
        self.cap = None
        
        self.setup_ui()

    def setup_ui(self):
        # 控制面板
        control_frame = ttk.LabelFrame(self.root, text="操作控制台", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # -> 模式一：实时摄像头
        ttk.Label(control_frame, text="模式一：摄像签到").pack(side=tk.LEFT, padx=(5, 10))
        ttk.Button(control_frame, text="打开摄像头", command=self.start_camera).pack(side=tk.LEFT, padx=5)
        # 签到按钮（未开摄像头时禁用）
        self.btn_snap = ttk.Button(control_frame, text="拍摄并签到", command=self.snap_and_check_in, state=tk.DISABLED)
        self.btn_snap.pack(side=tk.LEFT, padx=5)
        
        # -> 分隔符
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=20)
        
        # -> 模式二：照片上传
        ttk.Label(control_frame, text="模式二：照片签到").pack(side=tk.LEFT, padx=(10, 10))
        ttk.Button(control_frame, text="上传合影打卡", command=self.upload_and_check_in).pack(side=tk.LEFT, padx=5)
        
        # -> 底库更新 (靠右对齐)
        ttk.Button(control_frame, text="更新人员底库", command=self.update_db).pack(side=tk.RIGHT, padx=5)
        
        # 底部状态栏
        self.status_var = tk.StringVar(value="系统就绪。请选择上方打卡模式开始。")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 视频/图像显示区
        self.video_frame = ttk.Label(self.root, text="实时监控画面将显示在此处", relief=tk.RIDGE, anchor=tk.CENTER)
        self.video_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def start_camera(self):
        if self.is_camera_on: return
        try:
            self.cap = cv2.VideoCapture(CONFIG['camera_id'])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['resolution'][0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['resolution'][1])
            self.is_camera_on = True
            self.btn_snap.config(state=tk.NORMAL)
            self.status_var.set("摄像头运行中。请点击“拍摄并签到”进行打卡。")
            self.update_frame()
        except Exception as e:
            messagebox.showerror("错误", f"无法打开摄像头: {e}")

    def update_frame(self):
        if self.is_camera_on and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((960, 540), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)
            self.root.after(30, self.update_frame)

    def stop_camera(self):
        if self.is_camera_on:
            self.is_camera_on = False
            if self.cap: self.cap.release()
            self.video_frame.configure(image='')
            self.video_frame.configure(text="摄像头已关闭。")
            self.btn_snap.config(state=tk.DISABLED)
            self.status_var.set("摄像头已关闭。")

    # --- 核心逻辑：处理单帧图像 ---
    def process_image(self, frame):
        """统一处理识别、日志保存、证据留存与结果展示"""
        self.status_var.set("正在使用 AI 识别图像... 请稍候。")
        self.root.update()
        
        # 1. 人脸识别
        results = self.system.recognize(frame)
        
        present_names = set()
        frame_to_save = frame.copy()
        
        # 2. 绘制包含置信度的证据照片
        for (bbox, name, score) in results:
            x1, y1, x2, y2 = bbox
            color_cv = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            color_pil = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
            
            cv2.rectangle(frame_to_save, (x1, y1), (x2, y2), color_cv, 2)
            text_full = f"{name} ({score:.2f})"
            # 调用底层绘制中文的函数
            frame_to_save = self.draw_chinese_text(frame_to_save, text_full, (x1, y1 - 25), color_pil, 20)
            
            if name != "Unknown":
                present_names.add(name)

        # 3. 保存 CSV 日志与现场照片
        self.system.save_log(list(present_names))
        self.save_evidence_photo(frame_to_save)
        
        self.status_var.set(f"签到处理完毕。成功识别并打卡 {len(present_names)} 人。")
        
        # 4. 弹出签到结果名单窗口
        self.show_result_window(present_names, len(results))

    # 模式一触发：拍摄打卡
    def snap_and_check_in(self):
        if not self.is_camera_on or not hasattr(self, 'current_frame'):
            return
        self.process_image(self.current_frame.copy())

    # 模式二触发：上传照片打卡
    def upload_and_check_in(self):
        file_path = filedialog.askopenfilename(
            title="请选择要用于签到的会议合影",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            filename = os.path.basename(file_path)
            self.status_var.set(f"正在加载图片: {filename} ...")
            self.root.update()
            try:
                # 使用 imdecode 完美支持中文路径图片的读取
                img_data = np.fromfile(file_path, dtype=np.uint8)
                frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.process_image(frame)
                else:
                    messagebox.showerror("错误", "图片解码失败，请检查文件格式。")
                    self.status_var.set("系统就绪。")
            except Exception as e:
                messagebox.showerror("读取错误", f"无法加载文件: {e}")
                self.status_var.set("系统就绪。")

    # 独立的结果展示弹窗
    def show_result_window(self, present_names, total_detected):
        result_win = tk.Toplevel(self.root)
        result_win.title("会议签到明细")
        result_win.geometry("400x500")
        
        # 顶部汇总信息
        title_txt = f"画面中检测到人脸总数: {total_detected}\n成功匹配并签到人数: {len(present_names)}"
        ttk.Label(result_win, text=title_txt, font=("微软雅黑", 12, "bold"), padding=15, justify=tk.CENTER).pack()
        
        # 核心名单表格 (Treeview)
        columns = ("name", "status")
        tree = ttk.Treeview(result_win, columns=columns, show="headings", selectmode="none")
        
        tree.heading("name", text="参会人员姓名", anchor=tk.CENTER)
        tree.heading("status", text="签到状态", anchor=tk.CENTER)
        
        tree.column("name", width=200, anchor=tk.CENTER)
        tree.column("status", width=150, anchor=tk.CENTER)
        
        tree.pack(expand=True, fill=tk.BOTH, padx=20, pady=5)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(result_win, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 填充表格数据
        if not present_names:
             tree.insert("", tk.END, values=("未能识别出库中人员", "❌ 未签到"), tags=('unknown',))
        else:
            sorted_names = sorted(list(present_names))
            for name in sorted_names:
                tree.insert("", tk.END, values=(name, "✅ 已打卡"), tags=('present',))
        
        # 设置状态颜色
        tree.tag_configure('present', foreground='green')
        tree.tag_configure('unknown', foreground='red')

        # 底部关闭按钮
        ttk.Button(result_win, text="关闭窗口", command=result_win.destroy).pack(pady=15)

    # --- 辅助功能 ---
    def save_evidence_photo(self, frame):
        try:
            photo_dir = os.path.join(base_path, "attendance_photos")
            if not os.path.exists(photo_dir): os.makedirs(photo_dir)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filepath = os.path.join(photo_dir, f"打卡证据_{timestamp}.jpg") # 文件名也加上中文
            success, img_encoded = cv2.imencode('.jpg', frame)
            if success: img_encoded.tofile(filepath)
        except Exception as e:
            print(f"照片保存异常: {e}")

    def draw_chinese_text(self, img_cv, text, position, color=(0, 255, 0), text_size=20):
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("msyh.ttc", text_size) # 微软雅黑
        except:
            font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def update_db(self):
        self.status_var.set("正在提取特征并更新底库，这可能需要一点时间...")
        self.root.update()
        count, msg = self.system.register_faces()
        messagebox.showinfo("底库更新完毕", f"{msg}\n当前库中录入总人数: {count} 人")
        self.status_var.set("底库更新完毕，系统就绪。")

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    print("Initializing system...")
    # Ensure high DPI awareness for Windows (makes text clearer)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    face_sys = FaceSystem()
    root = tk.Tk()
    app = AttendanceApp(root, face_sys)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()