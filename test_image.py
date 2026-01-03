import cv2
import numpy as np
from main import MeetingAttendanceSystem # 假设主程序文件名叫 main.py

def test_static_image(image_path):
    # 初始化系统
    system = MeetingAttendanceSystem()
    
    # 检查数据库是否建立成功
    if system.known_embeddings is None:
        print("[ERROR] 特征库为空！请先在 database_images 放入照片并重新运行。")
        return

    # 读取测试图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] 无法读取图片: {image_path}")
        return

    print(f"[INFO] 正在识别图片: {image_path} ...")
    
    # 识别
    results = system.recognize_frame(frame)
    
    # 画框并保存
    system.draw_and_save(frame, results)
    
    # 因为不是实时视频，这里直接保存结果图而不是 cv2.imshow
    cv2.imwrite("result_output.jpg", frame)
    print(f"[SUCCESS] 结果已保存为 result_output.jpg")
    
    # 如果在有界面的环境，可以弹窗显示
    try:
        cv2.imshow("Test Result", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass

if __name__ == "__main__":
    # 确保你放了一张集体照叫 test_meeting.jpg
    test_static_image("test_meeting.jpg")
