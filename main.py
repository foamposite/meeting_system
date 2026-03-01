import os
import sys
import pickle
import datetime
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

# --- 🟢 全局轻量级库 (界面秒开必备) ---
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

# --- 必须保留的 NullWriter (防止无黑框模式闪退) ---
class NullWriter:
    def write(self, text): pass
    def flush(self): pass

if sys.stdout is None: sys.stdout = NullWriter()
if sys.stderr is None: sys.stderr = NullWriter()

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

base_path = get_base_path()

# --- 1. 配置与状态管理 ---
CONFIG_FILE = os.path.join(base_path, 'config.json')
CLASSES_FILE = os.path.join(base_path, 'classes.json')

def load_classes():
    """读取本地存储的班级名单字典"""
    if os.path.exists(CLASSES_FILE):
        try:
            with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: pass
    return {}

def save_classes(classes_dict):
    """保存班级名单到本地"""
    with open(CLASSES_FILE, 'w', encoding='utf-8') as f:
        json.dump(classes_dict, f, ensure_ascii=False, indent=4)

# --- 2. 核心算法类 (支持按班级划分) ---
class FaceSystem:
    def __init__(self):
        model_root = os.path.join(base_path, 'models')
        # 初始化 AI
        self.app = FaceAnalysis(name='buffalo_l', root=model_root, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.active_class = None
        self.active_roster = []
        self.known_embeddings = None
        self.known_names = []
        
    def set_class(self, class_name, roster):
        """切换当前激活的班级，并加载对应的底库"""
        self.active_class = class_name
        self.active_roster = roster
        self.db_file = os.path.join(base_path, 'models', f"db_{class_name}.pkl")
        self.load_database()

    def l2_normalize(self, x):
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norm + 1e-10)

    def load_database(self):
        self.known_embeddings = None
        self.known_names = []
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_names = data['names']
                    self.known_embeddings = data['embeddings']
            except: pass

    def register_faces(self):
        """按照班级名单提取照片特征，并进行严格核对"""
        if not self.active_class:
            return False, "请先选择一个班级！"
            
        # 班级照片文件夹路径
        class_img_folder = os.path.join(base_path, "database_images", self.active_class)
        if not os.path.exists(class_img_folder):
            os.makedirs(class_img_folder)
            return False, f"未找到班级照片文件夹！已自动创建：\ndatabase_images/{self.active_class}\n请将带有姓名的照片放入此文件夹。"

        embeddings_list, names_list = [], []
        found_names = set()
        
        for filename in os.listdir(class_img_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(class_img_folder, filename)
                try:
                    img_data = np.fromfile(img_path, dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if img is None: continue
                    faces = self.app.get(img)
                    if len(faces) == 1:
                        embeddings_list.append(faces[0].embedding)
                        names_list.append(name)
                        found_names.add(name)
                except: continue
        
        if embeddings_list:
            self.known_embeddings = self.l2_normalize(np.array(embeddings_list))
            self.known_names = names_list
            with open(self.db_file, 'wb') as f:
                pickle.dump({'names': names_list, 'embeddings': self.known_embeddings}, f)
            
            # --- 进行名单核对对账 ---
            total_roster = len(self.active_roster)
            total_found = len(found_names)
            missing_names = list(set(self.active_roster) - found_names)
            unregistered_names = list(found_names - set(self.active_roster)) # 库里有照片但不在名单里的人
            
            report = f"班级总人数: {total_roster} 人\n已录入照片: {total_found} 人\n"
            if missing_names:
                report += f"\n【⚠️ 缺失照片人员 ({len(missing_names)}人)】:\n" + "，".join(missing_names)
            if unregistered_names:
                report += f"\n\n【⚠️ 编外人员照片 ({len(unregistered_names)}人)】:\n" + "，".join(unregistered_names)
            if not missing_names and not unregistered_names:
                report += "\n【✅ 完美匹配】：所有人员照片均已齐全！"
                
            return True, report
        return False, "未能在文件夹中检测到任何有效的单人照片。"

    def recognize(self, frame):
        faces = self.app.get(frame)
        results = []
        if len(faces) > 0 and self.known_embeddings is not None:
            current_embs = self.l2_normalize(np.array([face.embedding for face in faces]))
            sim_matrix = np.dot(current_embs, self.known_embeddings.T)
            for i, face in enumerate(faces):
                max_idx = np.argmax(sim_matrix[i])
                max_score = sim_matrix[i][max_idx]
                name = self.known_names[max_idx] if max_score > 0.5 else "Unknown" # 阈值可调
                results.append((face.bbox.astype(int), name, max_score))
        return results

    def save_excel_log(self, present_names):
        """将打卡结果保存到结构化的 Excel 文件中"""
        if not self.active_class or not self.active_roster:
            return
            
        try:
            # 局部懒加载 pandas，保证不拖慢软件启动速度
            import pandas as pd
        except ImportError:
            print("缺少 pandas 库，无法导出 Excel")
            return

        # 计算缺勤人员
        absent_names = list(set(self.active_roster) - set(present_names))
        
        timestamp_full = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 表单名：年-月-日-小时
        sheet_name = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        
        data = []
        for name in self.active_roster:
            if name in present_names:
                data.append({"姓名": name, "状态": "✅ 已签到", "打卡时间": timestamp_full})
            else:
                data.append({"姓名": name, "状态": "❌ 缺勤", "打卡时间": ""})
                
        # 编外人员（不在名单但识别出来了）
        extra_names = set(present_names) - set(self.active_roster)
        for name in extra_names:
            if name != "Unknown":
                data.append({"姓名": name, "状态": "⚠️ 编外签到", "打卡时间": timestamp_full})
                
        df = pd.DataFrame(data)
        
        # 确保目录存在
        record_dir = os.path.join(base_path, "打卡记录", self.active_class)
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
            
        excel_path = os.path.join(record_dir, f"{self.active_class}打卡记录.xlsx")
        
        try:
            if os.path.exists(excel_path):
                # 追加到现有文件的新 sheet 中
                with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # 创建新文件
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        except Exception as e:
            messagebox.showerror("Excel 写入失败", f"保存报表失败！文件可能正被打开。\n报错: {e}")

# --- 3. UI 界面类 ---
class AttendanceApp:
    def __init__(self, root, system=None):
        self.root = root
        self.system = system 
        self.root.title("课堂智能签到系统")
        self.root.geometry("1050x750")
        
        self.is_camera_on = False
        self.cap = None
        self.is_paused = False
        self.available_cameras = []
        self.classes_dict = load_classes() # 读取班级数据
        
        self.setup_ui()
        self.refresh_camera_list()
        self.update_class_combo()
        
        if self.system is None:
            self.status_var.set("界面已启动，正在后台加载核心库，请稍候...")

    def setup_ui(self):
        # --- 顶部：班级管理模块 ---
        class_frame = ttk.LabelFrame(self.root, text="🏫 班级管理", padding="10")
        class_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(5,0))
        
        ttk.Label(class_frame, text="当前打卡班级：").pack(side=tk.LEFT, padx=5)
        self.class_combo = ttk.Combobox(class_frame, width=20, state="readonly")
        self.class_combo.pack(side=tk.LEFT, padx=5)
        self.class_combo.bind("<<ComboboxSelected>>", self.on_class_selected)
        
        ttk.Button(class_frame, text="➕ 导入名单并创建班级", command=self.import_class).pack(side=tk.LEFT, padx=10)
        ttk.Button(class_frame, text="🗑️ 删除当前班级", command=self.delete_class).pack(side=tk.LEFT, padx=5)
        
        # 提取到右侧：底库核对功能
        self.btn_update_db = ttk.Button(class_frame, text="🔄 更新并核对照片底库", command=self.update_db, state=tk.DISABLED)
        self.btn_update_db.pack(side=tk.RIGHT, padx=5)

        # --- 中间：操作控制台 ---
        control_frame = ttk.LabelFrame(self.root, text="⚙️ 打卡操作控制台", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_frame, text="📷 模式一：摄像打卡").pack(side=tk.LEFT, padx=(5, 5))
        self.camera_combo = ttk.Combobox(control_frame, width=12, state="readonly")
        self.camera_combo.pack(side=tk.LEFT, padx=5)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_change)
        
        ttk.Button(control_frame, text="↻", width=3, command=self.refresh_camera_list).pack(side=tk.LEFT, padx=2)
        self.btn_camera = ttk.Button(control_frame, text="打开摄像头", command=self.toggle_camera)
        self.btn_camera.pack(side=tk.LEFT, padx=5)
        
        self.btn_snap = ttk.Button(control_frame, text="📸 拍摄并签到", command=self.snap_and_check_in, state=tk.DISABLED)
        self.btn_snap.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=20)
        
        ttk.Label(control_frame, text="📁 模式二：照片打卡").pack(side=tk.LEFT, padx=(10, 10))
        self.btn_upload = ttk.Button(control_frame, text="上传合影打卡", command=self.upload_and_check_in, state=tk.DISABLED)
        self.btn_upload.pack(side=tk.LEFT, padx=5)
        
        # --- 底部：视频区与状态栏 ---
        self.video_frame = ttk.Label(self.root, text="请先在上方【选择或导入班级】，然后开启摄像头或上传照片", relief=tk.RIDGE, anchor=tk.CENTER)
        self.video_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.status_var = tk.StringVar(value="启动中...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # --- 班级管理功能 ---
    def update_class_combo(self):
        class_names = list(self.classes_dict.keys())
        self.class_combo['values'] = class_names
        if class_names:
            self.class_combo.current(0)
            self.on_class_selected(None)
        else:
            self.class_combo.set('')
            self.btn_update_db.config(state=tk.DISABLED)
            self.btn_snap.config(state=tk.DISABLED)
            self.btn_upload.config(state=tk.DISABLED)

    def on_class_selected(self, event):
        class_name = self.class_combo.get()
        if class_name and self.system:
            roster = self.classes_dict.get(class_name, [])
            self.system.set_class(class_name, roster)
            self.btn_update_db.config(state=tk.NORMAL)
            self.btn_upload.config(state=tk.NORMAL)
            if self.is_camera_on:
                self.btn_snap.config(state=tk.NORMAL)
            self.status_var.set(f"已切换至班级：{class_name} (名单共 {len(roster)} 人)")

    def import_class(self):
        file_path = filedialog.askopenfilename(title="选择包含名单的 Excel/CSV 文件", filetypes=[("表格文件", "*.xlsx *.xls *.csv")])
        if not file_path: return
        
        self.status_var.set("正在读取表格，请稍候...")
        self.root.update()
        
        try:
            # 懒加载 pandas
            import pandas as pd
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
                
            # 模糊匹配“姓名”列
            name_col = None
            for col in df.columns:
                if '姓名' in str(col) or 'name' in str(col).lower():
                    name_col = col
                    break
            
            if not name_col:
                # 如果没找到，默认取第一列
                name_col = df.columns[0]
                
            names_list = df[name_col].dropna().astype(str).str.strip().tolist()
            # 剔除空字符串
            names_list = [n for n in names_list if n]
            
            if not names_list:
                messagebox.showerror("错误", "提取名单失败：列中没有有效数据。")
                return
                
            # 弹窗询问班级名称
            default_name = os.path.splitext(os.path.basename(file_path))[0]
            class_name = simpledialog.askstring("定义班级名称", f"已成功提取 {len(names_list)} 人。\n请为该班级命名：", initialvalue=default_name)
            
            if class_name:
                self.classes_dict[class_name] = names_list
                save_classes(self.classes_dict)
                self.update_class_combo()
                self.class_combo.set(class_name)
                self.on_class_selected(None)
                
                # 自动创建照片文件夹
                img_folder = os.path.join(base_path, "database_images", class_name)
                if not os.path.exists(img_folder): os.makedirs(img_folder)
                
                messagebox.showinfo("导入成功", f"班级【{class_name}】创建成功！\n共导入 {len(names_list)} 名人员。\n\n下一步：请将这些人员的单人照片(以名字命名)放入文件夹：\ndatabase_images/{class_name}")
            
        except ImportError:
            messagebox.showerror("环境错误", "缺少必要的库！请在终端运行: pip install pandas openpyxl")
        except Exception as e:
            messagebox.showerror("读取错误", f"无法解析文件: {e}")
        finally:
            self.status_var.set("系统就绪。")

    def delete_class(self):
        class_name = self.class_combo.get()
        if not class_name: return
        
        if messagebox.askyesno("删除确认", f"确定要从系统中删除班级【{class_name}】吗？\n(注意：对应的考勤记录和照片文件夹不会被物理删除，仅从列表中移除)"):
            del self.classes_dict[class_name]
            save_classes(self.classes_dict)
            self.update_class_combo()

    def update_db(self):
        if not self.system or not self.class_combo.get(): return
        self.status_var.set("正在逐一核对名单与照片底库...")
        self.root.update()
        
        success, report_msg = self.system.register_faces()
        if success:
            messagebox.showinfo("底库核对报告", report_msg)
        else:
            messagebox.showwarning("核对失败", report_msg)
        self.status_var.set("底库更新核对完毕。")

    # --- 摄像头与打卡功能 ---
    def refresh_camera_list(self):
        self.available_cameras = []
        combo_list = []
        for i in range(4):
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
        if self.available_cameras: self.camera_combo.current(0)

    def on_camera_change(self, event):
        if self.is_camera_on:
            self.stop_camera()
            self.root.after(500, self.start_camera)

    def toggle_camera(self):
        if self.is_camera_on:
            self.stop_camera()
            self.btn_camera.config(text="打开摄像头")
        else:
            self.start_camera()
            if self.is_camera_on: self.btn_camera.config(text="关闭摄像头")

    def render_image_to_ui(self, frame_cv):
        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((960, 540), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk 
        self.video_frame.configure(image=imgtk)

    def start_camera(self):
        if not self.system: return
        try:
            selected_idx = self.camera_combo.current()
            camera_id = self.available_cameras[selected_idx] if selected_idx >= 0 else 0
        except: camera_id = 0

        try:
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(camera_id)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.is_camera_on = True
            self.is_paused = False 
            if self.class_combo.get():
                self.btn_snap.config(state=tk.NORMAL)
            self.status_var.set(f"正在使用摄像头 {camera_id} 运行中。")
            self.update_frame()
        except Exception as e:
            messagebox.showerror("错误", f"无法打开摄像头: {e}")
            self.is_camera_on = False

    def update_frame(self):
        if self.is_camera_on and self.cap.isOpened() and not self.is_paused:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.render_image_to_ui(frame)
        if self.is_camera_on: self.root.after(30, self.update_frame)

    def stop_camera(self):
        self.is_camera_on = False
        if self.cap: self.cap.release()
        self.video_frame.configure(image='')
        self.btn_snap.config(state=tk.DISABLED)
        self.status_var.set("摄像头已关闭。")

    def process_image(self, frame):
        if not self.class_combo.get():
            messagebox.showwarning("提示", "请先在上方选择或导入打卡班级！")
            return
            
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
        
        # 记录保存 (Excel 与 现场照片)
        self.system.save_excel_log(list(present_names))
        self.save_evidence_photo(frame_to_save)
        
        self.status_var.set(f"签到处理完毕。成功识别 {len(present_names)} 人。")
        self.show_result_window(present_names, len(results))

    def snap_and_check_in(self):
        if not self.is_camera_on or not hasattr(self, 'current_frame'): return
        self.process_image(self.current_frame.copy())

    def upload_and_check_in(self):
        file_path = filedialog.askopenfilename(title="选择合影", filetypes=[("图片", "*.jpg *.png *.bmp")])
        if file_path:
            self.status_var.set("正在加载图片...")
            self.root.update()
            try:
                frame = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    if self.is_camera_on: self.is_paused = True
                    self.current_frame = frame 
                    self.process_image(frame)
            except Exception as e:
                messagebox.showerror("错误", f"无法加载文件: {e}")

    def show_result_window(self, present_names, total_detected):
        result_win = tk.Toplevel(self.root)
        result_win.title(f"【{self.class_combo.get()}】 签到明细")
        result_win.geometry("400x550")
        
        def on_close():
            self.is_paused = False
            if self.is_camera_on: self.status_var.set("已恢复实时监控。")
            result_win.destroy()
            
        result_win.protocol("WM_DELETE_WINDOW", on_close)
        
        # 计算缺勤
        roster = self.classes_dict.get(self.class_combo.get(), [])
        absent_names = list(set(roster) - set(present_names))
        extra_names = list(set(present_names) - set(roster))
        
        title_txt = f"应到人数: {len(roster)} | 实到人数: {len(present_names)-len(extra_names)}\n缺勤人数: {len(absent_names)}"
        ttk.Label(result_win, text=title_txt, font=("微软雅黑", 12, "bold"), padding=10, justify=tk.CENTER).pack()
        
        tree = ttk.Treeview(result_win, columns=("name", "status"), show="headings", selectmode="none")
        tree.heading("name", text="姓名"); tree.column("name", width=200, anchor=tk.CENTER)
        tree.heading("status", text="状态"); tree.column("status", width=150, anchor=tk.CENTER)
        tree.pack(expand=True, fill=tk.BOTH, padx=20, pady=5)
        
        # 先显示缺勤的 (红色)
        for name in sorted(absent_names):
            tree.insert("", tk.END, values=(name, "❌ 缺勤"), tags=('absent',))
        # 再显示签到的 (绿色)
        for name in sorted(list(set(present_names) & set(roster))):
            tree.insert("", tk.END, values=(name, "✅ 已打卡"), tags=('present',))
        # 最后显示编外人员 (黄色/橙色)
        for name in extra_names:
            if name != "Unknown":
                tree.insert("", tk.END, values=(name, "⚠️ 编外识别"), tags=('extra',))
        
        tree.tag_configure('present', foreground='green')
        tree.tag_configure('absent', foreground='red')
        tree.tag_configure('extra', foreground='orange')
        
        ttk.Button(result_win, text="关闭窗口并继续", command=on_close).pack(pady=10)

    def save_evidence_photo(self, frame):
        class_name = self.class_combo.get()
        if not class_name: return
        try:
            record_dir = os.path.join(base_path, "打卡记录", class_name)
            if not os.path.exists(record_dir): os.makedirs(record_dir)
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filepath = os.path.join(record_dir, f"{class_name}_{timestamp}_打卡照片.jpg")
            cv2.imencode('.jpg', frame)[1].tofile(filepath)
        except Exception as e: 
            print("保存图片失败:", e)

    def draw_chinese_text(self, img_cv, text, position, color=(0, 255, 0), text_size=20):
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try: font = ImageFont.truetype("msyh.ttc", text_size)
        except: font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

# --- 4. 程序入口 ---
if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass

    root = tk.Tk()
    app = AttendanceApp(root, system=None) 
    root.update()
    
    def load_ai_thread():
        try:
            app.status_var.set("正在导入视觉算法核心 (这可能需要几秒钟)...")
            global insightface, FaceAnalysis
            import insightface
            from insightface.app import FaceAnalysis
            
            app.status_var.set("正在初始化 AI 模型，即将完成...")
            face_sys = FaceSystem() 
            root.after(0, on_ai_loaded, face_sys) 
        except Exception as e:
            root.after(0, on_ai_error, str(e))

    def on_ai_loaded(face_sys):
        app.system = face_sys
        # 如果启动时已有班级，自动激活
        app.on_class_selected(None)
        app.status_var.set("AI 模型加载完毕，系统就绪。请选择班级后进行操作。")

    def on_ai_error(err_msg):
        app.status_var.set("AI 模型加载失败！")
        messagebox.showerror("致命错误", f"模型加载失败，请检查 models 文件夹。\n{err_msg}")

    threading.Thread(target=load_ai_thread, daemon=True).start()
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()