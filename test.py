from PIL import Image
import cv2
import os
import sys
import torch
import os

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff, Audio2CoeffV2
from src.utils.init_path import init_path
from src.generate_batch import get_data

# IMAGE_PATH = "/home/t-ibayashi/Workspace/ax/repos/SadTalker/examples/source_image/art_5.png"
# AUDIO_PATH = "/home/t-ibayashi/Workspace/ax/repos/SadTalker/examples/driven_audio/imagine.wav"

IMAGE_PATH = "/content/SadTalker/examples/source_image/art_5.png"
AUDIO_PATH = "/content/SadTalker/examples/driven_audio/imagine.wav"

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

    result = preprocess_model.generate(IMAGE_PATH, "test_results", "resize", source_image_flag=True, pic_size=256)
    import pdb; pdb.set_trace()

def test_audio2coeff_generate():
    input_image_path = IMAGE_PATH
    input_audio_path = AUDIO_PATH
    output_dir = "./test_results"
    ref_eyeblink = None
    ref_pose = None
    current_root_path = os.path.split(sys.argv[0])[0]
    checkpoint_dir = './checkpoints'
    SIZE = 256
    PREPROCESS = 'crop'
    device = 'cpu'
    still = False
    ref_eyeblink_coeff_path = None
    save_dir = "test_results"
    pose_style = 0
    ref_pose_coeff_path = None
    sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), SIZE, False, PREPROCESS)

    preprocess_model = CropAndExtract(sadtalker_paths, device)
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        input_image_path,
        save_dir,
        "resize",
        source_image_flag=True,
        pic_size=SIZE,
    )

    batch = get_data(first_coeff_path, input_audio_path, device, ref_eyeblink_coeff_path, still=still)
    coeff_save_dir = os.path.join(save_dir, "0"),

    set_seed()
    model = Audio2CoeffV2(sadtalker_paths,  device)
    with torch.no_grad():
        exp_pred = model.audio2exp_model.forward(batch['indiv_mels'], batch['ref'],  batch['ratio_gt'])
        import pdb; pdb.set_trace()
        # pose_style = 0
        # batch['class'] = torch.LongTensor([pose_style]).to(model.device)
        # results_dict_pose = model.audio2pose_model.test(batch)
        # import pdb; pdb.set_trace()
        pass

def test_audio2coeff():
    input_image_path = IMAGE_PATH
    input_audio_path = AUDIO_PATH
    output_dir = "./test_results"
    ref_eyeblink = None
    ref_pose = None
    current_root_path = os.path.split(sys.argv[0])[0]
    checkpoint_dir = './checkpoints'
    SIZE = 256
    PREPROCESS = 'crop'
    device = 'cpu'
    still = False
    ref_eyeblink_coeff_path = None
    save_dir = "test_results"
    pose_style = 0
    ref_pose_coeff_path = None
    sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), SIZE, False, PREPROCESS)

    preprocess_model = CropAndExtract(sadtalker_paths, device)
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        input_image_path,
        save_dir,
        "resize",
        source_image_flag=True,
        pic_size=SIZE,
    )
    batch = get_data(first_coeff_path, input_audio_path, device, ref_eyeblink_coeff_path, still=still)

    # Audio2Exp.audio2exp_model.netGの出力を変更前後で比較
    # model = Audio2Coeff(sadtalker_paths,  device)
    # submodel = model.audio2exp_model.netG

    """
    2つのモデルの出力を変更前後で比較
    """

    # set_seed()
    # model_1 = Audio2Coeff(sadtalker_paths,  device)
    # out_1 = model_1.generate(batch, os.path.join(save_dir, "1"), pose_style, ref_pose_coeff_path)

    set_seed()
    model_2 = Audio2CoeffV2(sadtalker_paths,  device)
    out_2 = model_2.generate(batch, os.path.join(save_dir, "2"), pose_style, ref_pose_coeff_path)
    
def test_main():
    input_image_path = IMAGE_PATH
    input_audio_path = AUDIO_PATH
    output_dir = "./test_results"
    ref_eyeblink = None
    ref_pose = None
    current_root_path = os.path.split(sys.argv[0])[0]
    checkpoint_dir = './checkpoints'
    SIZE = 256
    PREPROCESS = 'crop'
    device = 'cpu'
    still = False
    ref_eyeblink_coeff_path = None
    save_dir = "test_results"
    pose_style = 0
    ref_pose_coeff_path = None
    sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), SIZE, False, PREPROCESS)

    preprocess_model = CropAndExtract(sadtalker_paths, device)
    audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
    # audio_to_coeff = Audio2CoeffV2(sadtalker_paths,  device)
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        input_image_path,
        save_dir,
        "resize",
        source_image_flag=True,
        pic_size=SIZE,
    )
    batch = get_data(first_coeff_path, input_audio_path, device, ref_eyeblink_coeff_path, still=still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
    import pdb; pdb.set_trace()

def set_seed():
    import random
    random.seed(314)

    import torch
    torch.manual_seed(0)

    import numpy as np
    np.random.seed(0)

set_seed()
# os.remove("/home/t-ibayashi/Workspace/ax/repos/SadTalker/test_results/art_5_landmarks.txt")
# test_FaceDetector()
# test_keypoint_extractor()
# test_preprocess()
test_audio2coeff_generate()
# test_audio2coeff()
# test_main()
