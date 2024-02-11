import os
import sys
import torch

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff, Audio2CoeffV2
from src.utils.init_path import init_path

SIZE = 256
PREPROCESS = 'crop'
current_root_path = os.path.split(sys.argv[0])[0]
checkpoint_dir = './checkpoints'
# if torch.cuda.is_available():
#     device = "cuda"
# else:
device = "cpu"
sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), SIZE, False, PREPROCESS)

if False:
  """
  CropAndExtract
  """
  preprocess_model = CropAndExtract(sadtalker_paths, device)

  # CropAndExtract.net_recon
  target = preprocess_model.net_recon
  # 固定されている: https://github.com/OpenTalker/SadTalker/blob/ae69a4c57c9838370643dd2c9b7d1f6ba16a54d8/src/utils/preprocess.py#L145
  input_size = 224
  torch.onnx.export(
      model=target,
      f="imageToCoeff.onnx",
      args=(torch.randn(1, 3, input_size, input_size),),
      export_params=True,
      opset_version=10,
      verbose=False,
      input_names = ["image"],
      output_names=["coeffitients"],
  )

  # keypoint extractor: face detector
  target = preprocess_model.propress.predictor.det_net
  input_size = 256
  torch.onnx.export(
      model=target,
      f="faceDetector.onnx",
      args=(torch.randn(1, 3, input_size, input_size),),
      export_params=True,
      opset_version=10,
      verbose=False,
      input_names=["image"],
      output_names=["location", "confidence", "landmarks"],
      dynamic_axes={ "image": {0: "batch_size", 2: "height", 3:"width"}},
  )

  # keypoint extractor: face aligner
  target = preprocess_model.propress.predictor.detector
  input_size = 256
  torch.onnx.export(
      model=target,
      f="faceAligner.onnx",
      args=(torch.randn(1, 3, input_size, input_size),),
      export_params=True,
      opset_version=10,
      verbose=False,
      input_names=["image"],
      # output_names=["location", "confidence", "landmarks"],
      # dynamic_axes={ "image": {0: "height", 1:"width"}},
  )


if True:
    """
    Audio2Coeff
    """
    audio_to_coeff = Audio2CoeffV2(sadtalker_paths, device)

    print("Audio2Coeff.audio2exp_model")
    target = audio_to_coeff.audio2exp_model
    torch.onnx.export(
        model=target,
        f="audio2Exp.onnx",
        args=(torch.randn(1, 140, 1, 80, 16), torch.randn(1, 140, 70), torch.randn(1, 140, 1), ),
        export_params=True,
        opset_version=10,
        verbose=False,
        input_names=["mel_input", "ref", "ratio"],
        output_names=["exp"],
    )

    # Audio2Coeff.audio2Pose.audioEncoder
    print("Audio2Coeff.audio2pose_model.audio_encoder")
    target = audio_to_coeff.audio2pose_model.audio_encoder
    torch.onnx.export(
        model=target,
        f="audio2Pose_audioEncoder.onnx",
        args=(torch.randn(1, 32, 1, 80, 16)),
        export_params=True,
        opset_version=10,
        verbose=False,
        input_names=["melspectrograms"],
        output_names=["audio_emb"],
    )

    # Audio2Coeff.audio2Pose.netG
    target = audio_to_coeff.audio2pose_model.netG
    torch.onnx.export(
        model=target,
        f="audio2Pose_netG.onnx",
        args=(
            torch.randn(1, 64),
            torch.randn(1),
            torch.randn(1, 6),
            torch.randn(1, 32, 512),
        ),
        export_params=True,
        opset_version=10,
        verbose=False,
        input_names=["z", "class", "ref", "audio_emb"],
        output_names=["pose_motion_pred"],
    )

    

print("success")









