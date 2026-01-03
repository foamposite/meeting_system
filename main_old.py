import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import pickle
import datetime
from sklearn.preprocessing import normalize
import sys
import os

# 1. 定义一个函数来获取真实的运行路径
def get_base_path():
    if getattr(sys, 'frozen', False):
        # 如果是打包后的 exe 运行，取 exe 所在目录
        return os.path.dirname(sys.executable)
    else:
        # 如果是脚本运行，取脚本所在目录
        return os.path.dirname(os.path.abspath(__file__))

base_path = get_base_path()

# 2. 修改类初始化中的路径 (拼接 base_path)
class MeetingAttendanceSystem:
    def __init__(self, db_folder='database_images', db_file='face_db.pkl'):
        # 强制将路径指向 EXE 同级目录
        self.db_folder = os.path.join(base_path, db_folder)
        self.db_file = os.path.join(base_path, db_file)
        
        # --- 关键修改：指定 InsightFace 模型根目录 ---
        # 我们希望模型文件夹就在 exe 旁边，而不是在用户的主目录
        # 这样拷贝给别人时，模型也能一起带走
        self.app = FaceAnalysis(
            name='buffalo_l', 
            root=base_path,  # <--- 新增：强制指定模型根目录为当前目录
            providers=['CPUExecutionProvider']
        )

class MeetingAttendanceSystem:
    def __init__(self, db_folder='database_images', db_file='face_db.pkl'):
        self.db_folder = db_folder
        self.db_file = db_file
        
        # --- 初始化 InsightFace ---
        # model='buffalo_l': 精度优先模型 (L代表Large)
        # providers: 优先使用CUDA，如果没有则使用CPU
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        # prepare: ctx_id=0 表示使用第一个GPU (-1为CPU), det_size=(640, 640) 是检测分辨率
        # 注意：为了检测会议室后排小脸，建议将 det_size 设置得大一些，或者设为 None (自动适应)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # 内存中存储的特征库
        self.known_embeddings = None
        self.known_names = []
        
        # 加载数据库
        self.load_database()

    def load_database(self):
        """加载或创建人脸特征数据库"""
        if os.path.exists(self.db_file):
            print(f"[INFO] 正在加载现有特征库: {self.db_file}")
            with open(self.db_file, 'rb') as f:
                data = pickle.load(f)
                self.known_names = data['names']
                self.known_embeddings = data['embeddings']
        else:
            print("[INFO] 未找到特征库，准备新建...")
            self.register_faces()

    def register_faces(self):
        """遍历图片文件夹，提取特征并建库"""
        if not os.path.exists(self.db_folder):
            os.makedirs(self.db_folder)
            print(f"[WARN] 文件夹 {self.db_folder} 不存在，已创建。请放入员工照片后重新运行。")
            return

        embeddings_list = []
        names_list = []

        print("[INFO] 开始处理底库图片...")
        for filename in os.listdir(self.db_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(self.db_folder, filename)
                
                # 读取图片
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # 检测人脸
                faces = self.app.get(img)
                
                if len(faces) != 1:
                    print(f"[SKIP] {filename}: 图片中未检测到人脸或检测到多张人脸，跳过。")
                    continue
                
                # InsightFace提取的特征通常是未归一化的，虽然ArcFace输出是归一化的，
                # 但为了保险，我们手动做一次 sklearn 的 normalize
                embedding = faces[0].embedding
                embeddings_list.append(embedding)
                names_list.append(name)
                print(f"[SUCCESS] 已录入: {name}")

        if embeddings_list:
            # 转换为 Numpy 矩阵 (N, 512)
            self.known_embeddings = np.array(embeddings_list)
            # 归一化特征向量（对于余弦相似度至关重要）
            self.known_embeddings = normalize(self.known_embeddings)
            self.known_names = names_list
            
            # 保存到文件
            with open(self.db_file, 'wb') as f:
                pickle.dump({'names': names_list, 'embeddings': self.known_embeddings}, f)
            print(f"[INFO] 数据库构建完成，共 {len(names_list)} 人。")
        else:
            print("[ERROR] 未能成功录入任何图片。")

    def recognize_frame(self, frame, threshold=0.4):
        """
        处理单帧画面：检测 -> 识别 -> 标注
        threshold: 相似度阈值 (0.4-0.6之间，InsightFace通常0.5较好)
        """
        # 1. 检测当前画面所有人脸
        faces = self.app.get(frame)
        
        results = []
        
        if len(faces) > 0 and self.known_embeddings is not None:
            # 获取现场所有人脸特征矩阵 (M, 512)
            current_embeddings = np.array([face.embedding for face in faces])
            current_embeddings = normalize(current_embeddings)
            
            # 2. 矩阵乘法计算余弦相似度
            # (M, 512) x (N, 512).T = (M, N)
            # sim_matrix[i][j] 表示 第i个现场人脸 与 第j个库中人脸 的相似度
            sim_matrix = np.dot(current_embeddings, self.known_embeddings.T)
            
            # 3. 找到匹配结果
            for i, face in enumerate(faces):
                # 找到该人脸与底库最大的相似度
                max_score = np.max(sim_matrix[i])
                max_idx = np.argmax(sim_matrix[i])
                
                name = "Unknown"
                if max_score > threshold:
                    name = self.known_names[max_idx]
                
                # 记录结果：(位置框, 名字, 相似度)
                bbox = face.bbox.astype(int)
                results.append((bbox, name, max_score))
                
        return results

    def run_camera(self):
        """启动摄像头实时签到"""
        cap = cv2.VideoCapture(0) # 0为默认摄像头，外接USB摄像头可能是1
        # 设置为高清分辨率（重要：提高远距离识别率）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        print("[INFO] 摄像头已启动。按 'q' 键拍摄并签到，按 'ESC' 退出程序。")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 为了流畅，平时只显示，不进行重型推理
            # 可以在这里做一些简单的resize显示
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press 'q' to Check-in", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Meeting Room Monitor', display_frame)
            
            key = cv2.waitKey(1)
            
            # 按 'q' 触发拍摄和识别
            if key & 0xFF == ord('q'):
                print("[INFO] 正在识别中...")
                results = self.recognize_frame(frame)
                
                # 绘制结果
                self.draw_and_save(frame, results)
                
                print("[INFO] 识别结束，请查看画面。按任意键继续...")
                cv2.waitKey(0) 
            
            # 按 ESC 退出
            elif key == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def draw_and_save(self, frame, results):
        """绘制方框并保存日志"""
        detected_people = set()
        
        for (bbox, name, score) in results:
            x1, y1, x2, y2 = bbox
            
            # 颜色：已知绿色，未知红色
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 写名字和置信度
            label = f"{name} ({score:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if name != "Unknown":
                detected_people.add(name)
        
        # 写入 CSV 日志
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if detected_people:
            with open('checkin_log.csv', 'a') as f:
                for p in detected_people:
                    f.write(f"{timestamp},{p},Present\n")
            print(f"[RESULT] 已签到: {list(detected_people)}")
        else:
            print("[RESULT] 未识别到已知人员。")
            
        cv2.imshow('Check-in Result', frame)

if __name__ == "__main__":
    # 实例化并运行
    # 第一次运行时，请确保 database_images 文件夹里有照片
    system = MeetingAttendanceSystem()
    system.run_camera()
