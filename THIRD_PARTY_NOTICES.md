# Third-party notices: the Motion Previs tab

The Motion Previs tab (`app/motion.py`, `app/motion_worker.py`, `app/static/motion.js`, `app/motion_templates/`,
`scripts/setup-motion.sh`) builds on the work below. None of it is bundled in this repository except where said:
packages and models are downloaded by `scripts/setup-motion.sh` (`oneshot-setup.sh --motion`) into
`$PREFIX/motion-venv` and `$PREFIX/motion-models`. The notices for the rest of the pack are in [NOTICE](NOTICE) and
[CREDITS.md](CREDITS.md).

## Motion Previs Studio, by Sam Wasserman

- Original: **Motion Previs Studio 4.1.0**, <https://github.com/wassermanproductions/motion-previs-studio>, read at
  commit `95e7d0ff1d4cc546f7eb09a74ccbd084988a19bc`.
- Copyright 2026 Sam Wasserman. Developed and created by Sam Wasserman for Wasserman Productions and Wasserman.ai.
- License: Apache License 2.0 (the text is this repository's [LICENSE](LICENSE)). Its NOTICE file is kept unchanged at
  [LICENSES/Motion-Previs-Studio-NOTICE](LICENSES/Motion-Previs-Studio-NOTICE).
- Windows support in that repository was contributed by Gumbii Digital; none of the Windows packaging is used here.

**What is taken, and that it was modified.** The analysis in this tab is a Python port of his: `app/motion_worker.py`
is a rewrite in Python of `src/lib/pose.ts`, `poseVideo.ts`, `openpose.ts`, `cameraMotion.ts`, `aiDepth.ts` and
`quality.ts`, and of the ffmpeg control passes and the production pack writer in `electron/main.cjs`. The pack's
file names, the layout of `bundle_manifest.json`, the schemas of `camera_motion.json`, `pose_landmarks.json` and
`openpose_keypoints.json`, the four reference modes, the wording of the prompt pack, shot bible and manifests, and the
three Blender import scripts in `app/motion_templates/` (copied from `electron/main.cjs` with a header added) are his.
The interface in `app/static/motion.js` follows his app's flow and shares no code with it.

**What differs from the original**

| Original | This port | Why |
|---|---|---|
| Electron + React/TypeScript desktop app | A tab of this web app; the analysis is one Python process per job | It runs beside the Studio's render queue, on a Mac or a DGX Spark, with no second app |
| MediaPipe Tasks Vision (WASM) in the renderer | The `mediapipe` Python package, same Pose Landmarker models and hashes | Same models, native speed |
| Pose model defaults to Lite | Defaults to Heavy (Lite and Full are offered) | Heavy tracks a 5 second shot in about a second here, and on the test clips Lite placed the shoulders well inside a coat |
| Camera solve written out by hand on a 192 pixel canvas: Shi-Tomasi corners, pyramidal Lucas-Kanade, seeded two-point RANSAC | OpenCV's `goodFeaturesToTrack`, `calcOpticalFlowPyrLK` (with a forward-backward check) and `estimateAffinePartial2D` (RANSAC, fixed seed) on a 384 pixel canvas | The same method on the library's tested routines; the same subject mask from the pose track, the same clamps and confidence |
| The shift of each step is read at the top-left corner of the frame | It is read at the centre of the frame | A push-in on the middle of the picture then has no pan or tilt in it |
| The roll of each step is the rotation of the similarity fit | Roll is kept only where both picture axes turn the same way | Travelling past near and far things shears the picture, and the fit read that as several degrees of roll that were not there |
| With no pose track, corners in the middle of the frame score half | Corners are taken from outside the middle first, then topped up | `goodFeaturesToTrack` takes a mask, not a weight |
| Depth-Anything-small through transformers.js (WebGPU or WASM), short side 518, at most 8 frames a second, levels set per frame | The same ONNX file (`Xenova/depth-anything-small-hf` at the pinned revision, 8-bit) through ONNX Runtime on the processor, long side 518, at the chosen sample rate, levels that follow the clip slowly | A quarter of the work for a control layer, no flicker between frames, and a layer smooth enough to hand on as a reference clip |
| A smaller source is enlarged to 1280 pixels | A smaller source keeps its size | The Studio's own clips are often 864 x 480 |
| Imports web video through yt-dlp; sends clips to Blockout; MCP control server; session restore; 3D preview | None of these | Not asked for here. No URL download exists in this tab |
| `pose_high_contrast.webm`, `openpose_pose.webm`, `ai_depth.webm` | Not written (their manifest entries are null) | They are the original's pre-v4 recorder output; the mp4 files are the same pictures |
| Any shot length | A shot is 2 to 15 seconds | The length of a reference clip MiniMax H3 Ref2VA accepts |
| - | `bundle_manifest.json` has one more key, `port`: what this is a port of, the camera move in words, per-stage timings, notes | The original's keys are unchanged and in its order |
| - | A "MiniMax H3" section at the end of `prompt_pack.md` with the camera move in words; `motion_reference()` | The hand-off this tab exists for |

## MediaPipe (Google)

- The `mediapipe` Python package 1.1.0, <https://github.com/google-ai-edge/mediapipe>. Apache License 2.0.
- Pose Landmarker models `pose_landmarker_lite.task`, `pose_landmarker_full.task`, `pose_landmarker_heavy.task`
  (float16, version 1), from `storage.googleapis.com/mediapipe-models`, with the SHA-256 hashes pinned in Motion Previs
  Studio's `ASSET_MANIFEST.json`. Apache License 2.0.

## Depth Anything

- **Depth Anything** (Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao; CVPR 2024),
  <https://github.com/LiheYoung/Depth-Anything>. Code: Apache License 2.0. The small model
  `LiheYoung/depth-anything-small-hf`: Apache License 2.0.
- The file used is its ONNX conversion `onnx/model_quantized.onnx` from `Xenova/depth-anything-small-hf` at revision
  `2e942621ab9f2371c1df9eb223291b5ac31475e6` (the repository and revision Motion Previs Studio pins), SHA-256
  `e4e2ffe9b0396ebd4a0c09b855762b186aab338e79f76531b09777aafc91db84`. That repository declares no license of its
  own; it is used under the Apache-2.0 license of the base model it names.

## OpenCV

- `opencv-contrib-python` 5.0.0.93 (the build MediaPipe requires), <https://github.com/opencv/opencv> and
  <https://github.com/opencv/opencv-python>. OpenCV: Apache License 2.0. The Python packaging: MIT. The wheels carry
  third-party libraries (among them FFmpeg libraries under LGPL-2.1-or-later) with their license files inside the wheel.
- Without the Motion Previs environment, the camera solve runs on the `opencv-python` already in the Mac Studio
  install, under the same licenses.

## ONNX Runtime (Microsoft)

- `onnxruntime` 1.31.0, <https://github.com/microsoft/onnxruntime>. MIT License.

## NumPy

- <https://numpy.org>. BSD 3-Clause License.

## FFmpeg

- The `ffmpeg` and `ffprobe` on the machine's PATH are run as separate programs (trim, control passes, encoding); none
  is bundled. FFmpeg is licensed under LGPL-2.1-or-later or GPL-2.0-or-later depending on how it was built.

This file is informational and does not replace the license texts shipped inside each package.
