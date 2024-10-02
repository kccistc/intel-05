# 필요한 라이브러리 import
import requests
import time
from pathlib import Path
import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
from notebook_utils import device_widget, download_file, load_image
import tkinter as tk
import threading
from typing import Optional
import openvino.properties as props

# 모델 다운로드 경로 및 설정
model_folder = Path("../model")
ir_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/"
ir_model_name_xml = "MiDaS_small.xml"
ir_model_name_bin = "MiDaS_small.bin"

# 모델 다운로드
download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=model_folder)
download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=model_folder)

# 모델 경로 설정
model_xml_path = model_folder / ir_model_name_xml

# 유틸리티 함수들 정의
def normalize_minmax(data):
    """Normalizes the values in `data` between 0 and 1"""
    return (data - data.min()) / (data.max() - data.min())

def convert_result_to_image(result, colormap="viridis"):
    """Convert network result of floating point numbers to an RGB image."""
    cmap = matplotlib.cm.get_cmap(colormap)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    result = result.astype(np.uint8)
    return result

def to_rgb(image_data) -> np.ndarray:
    """Convert image_data from BGR to RGB"""
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

def detect_closer_object(depth_map: np.ndarray, threshold: float = 0.2) -> bool:
    """Detect if any object in the scene is closer than the threshold."""
    if np.min(depth_map) < threshold:
        return True
    return False


# 장치 설정 및 모델 컴파일
device = device_widget()
cache_folder = Path("cache")
cache_folder.mkdir(exist_ok=True)

core = ov.Core()
core.set_property({props.cache_dir(): cache_folder})
model = core.read_model(model_xml_path)
compiled_model = core.compile_model(model=model, device_name=device.value)

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)

network_input_shape = list(input_key.shape)
network_image_height, network_image_width = network_input_shape[2:]


# Warning popup function
def popup_warning(message: str, duration: int = 5) -> None:
    root = tk.Tk()
    root.title("Warning")
    label = tk.Label(root, text=message, padx=20, pady=20)
    label.pack()
    root.after(duration * 1000, root.destroy)
    root.mainloop()


def show_warning() -> None:
    warning_thread = threading.Thread(
        target=popup_warning, args=("Object too close!", 3)
    )
    warning_thread.start()


# Depth normalization function
def normalize_minmax(data):
    return (data - data.min()) / (data.max() - data.min())


def convert_result_to_image(result, colormap="viridis"):
    cmap = matplotlib.cm.get_cmap(coqormap)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    result = result.astype(np.uint8)
    return result


def process_stream_with_warning():
    # 모델 경로가 이미 지정됨
    core = ov.Core()
    model = core.read_model(model_xml_path)
    compiled_model = core.compile_model(model=model, device_name="CPU")

    input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)

    # 웹캠 스트리밍 시작
    cap = cv2.VideoCapture(0)
    network_image_height, network_image_width = list(input_key.shape)[2:]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 크기 조정 및 모델 입력 준비
        resized_frame = cv2.resize(frame, (network_image_width, network_image_height))
        input_image = np.expand_dims(np.transpose(resized_frame, (2, 0, 1)), 0)

        # 모델 추론
        result = compiled_model([input_image])[output_key]
        depth_image = convert_result_to_image(result)
        depth_image_resized = cv2.resize(depth_image, (frame.shape[1], frame.shape[0]))

        # 원본 영상과 모노뎁스 결과 나란히 표시
        stacked_frame = np.hstack((frame, depth_image_resized))

        # 사용자가 너무 가까운지 확인하고 경고 창 띄우기
        min_depth_value = np.min(result)
        if min_depth_value < 0.5:  # '너무 가까움'을 나타내는 임계값
            show_warning()

        # 결과 화면 표시
        cv2.imshow("Webcam and Depth", stacked_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# 함수 실행
process_stream_with_warning()

