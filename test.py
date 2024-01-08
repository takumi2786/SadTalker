from PIL import Image
import cv2
import os
import sys
import torch

from src.utils.preprocess import CropAndExtract
from src.utils.init_path import init_path

IMAGE_PATH = "/home/t-ibayashi/Workspace/ax/repos/SadTalker/examples/source_image/art_5.png"



def test_FaceDetector():
    SIZE = 256
    PREPROCESS = 'crop'
    current_root_path = os.path.split(sys.argv[0])[0]
    checkpoint_dir = './checkpoints'
    device = 'cpu'
    sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), SIZE, False, PREPROCESS)
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    # face detector
    model = preprocess_model.propress.predictor.det_net

    pic_size=256
    full_frames = [cv2.imread(IMAGE_PATH)]
    x_full_frames= [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  for frame in full_frames]
    frames_pil = [Image.fromarray(cv2.resize(frame,(pic_size, pic_size))) for frame in x_full_frames]
    with torch.no_grad():
        model.detect_faces(frames_pil[0])

def test_keypoint_extractor():
    SIZE = 256
    PREPROCESS = 'crop'
    current_root_path = os.path.split(sys.argv[0])[0]
    checkpoint_dir = './checkpoints'
    device = 'cpu'
    sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), SIZE, False, PREPROCESS)
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    # keypoint_extractor
    model = preprocess_model.propress.predictor

    pic_size=256
    full_frames = [cv2.imread(IMAGE_PATH)]
    x_full_frames= [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  for frame in full_frames]
    frames_pil = [Image.fromarray(cv2.resize(frame,(pic_size, pic_size))) for frame in x_full_frames]

    model.extract_keypoint(frames_pil[0], name="./test_result")

def test_preprocess():
    SIZE = 256
    PREPROCESS = 'crop'
    current_root_path = os.path.split(sys.argv[0])[0]
    checkpoint_dir = './checkpoints'
    device = 'cpu'
    sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), SIZE, False, PREPROCESS)
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    preprocess_model.generate(IMAGE_PATH, "test_results", "resize", source_image_flag=True, pic_size=256)

# test_FaceDetector()
# test_keypoint_extractor()
test_preprocess()
