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
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont # 确保引入 ImageDraw, ImageFont
from sklearn.preprocessing import normalize
import platform

# --- 1. 路径与配置工具函数 ---

def get_base_path():
    """获取程序运行的基础路径（兼容 EXE 和 脚本模式）"""
    if getattr(sys, 'frozen', False):
        # 如果是打包后的 exe 运行，取 exe 所在目录
        return os.path.dirname(sys.executable)
    else:
        # 如果是脚本运行，取脚本所在目录
        return os.path.dirname(os.path.abspath(__file__))

base_path = get_base_path()

def load_config():
    config_path = os.path.join(base_path, 'config.json')
    # 默认配置
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
            print(f"配置文件读取失败: {e}，使用默认配置")
    return default_config

CONFIG = load_config()

# --- 2. 核心算法类 (后端逻辑) ---

class FaceSystem:
    def __init__(self):
        self.db_folder = os.path.join(base_path, CONFIG['db_folder'])
        self.db_file = os.path.join(base_path, CONFIG['db_file'])
        
        # 指定模型路径为 base_path 下的 models 文件夹
        # 这样打包时，只需把 models 文件夹放在 exe 旁边即可
        self.app = FaceAnalysis(
            name='buffalo_l', 
            root=os.path.join(base_path, 'models'), # 关键：模型路径外部化
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
        """重新扫描文件夹并生成底库 (支持中文路径)"""
        if not os.path.exists(self.db_folder):
            os.makedirs(self.db_folder)
            return 0, "文件夹不存在，已创建"

        embeddings_list = []
        names_list = []
        count = 0

        print(f"[INFO] 正在扫描: {self.db_folder}")

        for filename in os.listdir(self.db_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 1. 提取中文名字
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(self.db_folder, filename)
                
                # --- [修改点 1] 使用 numpy + imdecode 读取中文路径图片 ---
                try:
                    # np.fromfile 支持中文路径读取二进制流
                    img_data = np.fromfile(img_path, dtype=np.uint8)
                    # cv2.imdecode 将二进制流解码为 OpenCV 图像
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                except Exception as e:
                    print(f"读取失败 {filename}: {e}")
                    continue
                
                if img is None:
                    print(f"[WARN] 无法解码图片: {filename}")
                    continue
                
                # 2. 检测人脸
                faces = self.app.get(img)
                if len(faces) == 1:
                    embedding = faces[0].embedding
                    embeddings_list.append(embedding)
                    names_list.append(name)
                    count += 1
                    print(f"[SUCCESS] 录入: {name}")
                else:
                    print(f"[SKIP] {name}: 未检测到人脸或有多张脸")
        
        if embeddings_list:
            self.known_embeddings = normalize(np.array(embeddings_list))
            self.known_names = names_list
            with open(self.db_file, 'wb') as f:
                pickle.dump({'names': names_list, 'embeddings': self.known_embeddings}, f)
            return count, "底库更新成功"
        else:
            return 0, "未检测到有效图片"

    def recognize(self, frame):
        """识别单帧图片"""
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

# --- 3. UI 界面类 (前端显示) ---

class AttendanceApp:
    def __init__(self, root, system):
        self.root = root
        self.system = system
        self.root.title("会议签到系统 (研究生开发版)")
        self.root.geometry("1000x700")
        
        # 状态变量
        self.is_camera_on = False
        self.cap = None
        
        # --- 布局 ---
        # 1. 顶部控制栏
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(control_frame, text="打开摄像头", command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="拍摄并签到", command=self.check_in).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="更新底库", command=self.update_db).pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="就绪", foreground="blue")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # 2. 视频显示区域
        self.video_frame = ttk.Label(root)
        self.video_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # 3. 底部信息
        ttk.Label(root, text=f"配置: 阈值={CONFIG['threshold']} | 库位置={CONFIG['db_folder']}").pack(side=tk.BOTTOM, pady=5)

    def start_camera(self):
        if self.is_camera_on: return
        
        try:
            self.cap = cv2.VideoCapture(CONFIG['camera_id'])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['resolution'][0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['resolution'][1])
            self.is_camera_on = True
            self.status_label.config(text="摄像头运行中...")
            self.update_frame()
        except Exception as e:
            messagebox.showerror("错误", f"无法打开摄像头: {e}")

    def update_frame(self):
        if self.is_camera_on and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # CV2 (BGR) -> RGB -> PIL -> ImageTk
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 缩放以适应窗口 (可选)
                img = Image.fromarray(frame_rgb)
                
                # 保持比例缩放
                display_width = 960
                display_height = 540
                img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
                
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk # 防止垃圾回收
                self.video_frame.configure(image=imgtk)
                
                # 保存当前帧用于签到
                self.current_frame = frame 
                
            self.root.after(30, self.update_frame) # 33ms刷新一次 (~30fps)

    # --- 新增辅助函数：绘制中文 ---
    def draw_chinese_text(self, img_cv, text, position, color=(0, 255, 0), text_size=20):
        """
        img_cv: OpenCV图片 (BGR)
        text: 中文字符串
        position: (x, y)
        color: RGB颜色
        """
        # 1. OpenCV BGR -> PIL RGB
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 2. 加载中文字体 (Windows下默认使用微软雅黑)
        try:
            # 这里的路径是 Windows 的标准字体路径
            font = ImageFont.truetype("msyh.ttc", text_size)
        except:
            # 如果找不到微软雅黑，回退到默认字体（可能不支持中文）
            font = ImageFont.load_default()
        
        # 3. 绘制文字
        draw.text(position, text, font=font, fill=color)
        
        # 4. PIL RGB -> OpenCV BGR
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # --- 修改 check_in 函数 ---
    def check_in(self):
        if not self.is_camera_on or not hasattr(self, 'current_frame'):
            messagebox.showwarning("提示", "请先打开摄像头")
            return
            
        frame = self.current_frame.copy()
        
        self.status_label.config(text="正在识别中...")
        self.root.update()
        
        results = self.system.recognize(frame)
        
        names = []
        for (bbox, name, score) in results:
            x1, y1, x2, y2 = bbox
            # 颜色：已知绿色，未知红色
            # 注意：PIL 使用 RGB，OpenCV 使用 BGR。
            # 这里我们统一定义为 (0, 255, 0) 这种元组，但在 cv2.rectangle 需要 BGR
            color_cv = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # BGR: Green
            color_pil = (0, 255, 0) if name != "Unknown" else (255, 0, 0) # RGB: Green
            
            # 画框 (OpenCV画框比较快，继续用CV)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_cv, 2)
            
            # --- [修改点 2] 使用 PIL 绘制中文名字 ---
            # 原来的 cv2.putText 改为调用 self.draw_chinese_text
            display_text = f"{name} ({score:.2f})"
            frame = self.draw_chinese_text(frame, display_text, (x1, y1 - 25), color_pil, 20)

            if name != "Unknown": names.append(name)
        
        self.system.save_log(names)
        
        # 弹窗显示结果图
        result_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_win = tk.Toplevel(self.root)
        result_win.title(f"签到结果: {len(names)} 人成功")
        
        img = Image.fromarray(result_rgb)
        img = img.resize((800, 450))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl = tk.Label(result_win, image=imgtk)
        lbl.image = imgtk
        lbl.pack()
        
        self.status_label.config(text=f"签到完成: {','.join(names)}")

    def update_db(self):
        self.status_label.config(text="正在更新底库，请稍候...")
        self.root.update()
        count, msg = self.system.register_faces()
        messagebox.showinfo("底库更新", f"{msg}\n当前库中人数: {count}")
        self.status_label.config(text="底库更新完毕")

    def on_closing(self):
        if self.cap: self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    # 启动画面
    print("正在初始化系统...")
    face_sys = FaceSystem()
    
    root = tk.Tk()
    app = AttendanceApp(root, face_sys)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()