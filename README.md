# SpeechForensics (NeurIPS 2024)
This is a PyTorch implementation of [SpeechForensics: Audio-Visual Speech Representation Learning for Face Forgery Detection](https://openreview.net/forum?id=ZsS0megTsh).

![Model_Overview](docs/images/method.png)
## Setup
### Prerequisite
1. Install Python 3.10 and create a virtual environment named `.venv`.
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
2. `pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html`
   (Choose the suitable version for your machine.)
3. Clone this repository.
4. Install dependency packages via `pip install -r requirements.txt`.
5. Install AV-HuBert and face-alignment
   ```bash
   git submodule init
   git submodule update
   ```
6. Install Fairseq
   ```
   cd av_hubert
   git submodule init
   git submodule update
   cd fairseq
   pip install --editable ./
   ```
7. Install FFmpeg. We use version=4.2.2.
8. Put the `modification/retinaface` in `preprocessing/face-alignment/face_alignment/detection`
   ```bash
   cp -r modification/retinaface preprocessing/face-alignment/face_alignment/detection
   ```
   Copy the `modification/landmark_extract.py` to `preprocessing/face-alignment/landmark_extract.py`
   ```bash
   cp modification/landmark_extract.py preprocessing/face-alignment
   ```

### Prepare data
1. Follow the links below to download the datasets (you will be asked to fill out some forms before downloading):
    * [FaceForensics++](https://github.com/ondyari/FaceForensics) (Download the [audio](https://github.com/ondyari/FaceForensics/tree/master/dataset#audio) according to the youtube ids and extract audio clips using the frame numbers that can obtained by downloading the 'original_youtube_videos_info'. Alternatively, you may download it from this [link](https://drive.google.com/file/d/1Cu1JVmAoTbssAQ290DxNVnaBdRmEotWP/view?usp=sharing).)
    * [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb)
    * [KoDF](https://github.com/deepbrainai-research/kodf)
2. Place the videos in the corresponding directories.
   ```
   your_dataset_root
   |--FaceForensics
      |--c23
         |--Deepfakes
            |--videos
               |--000.mp4
   |--FakeAVCeleb
      |--videos
         |--RealVideo-RealAudio
            |--Africa
               |--man
   ```
    * The directory structure of FaceForensics++: `your_dataset_root/FaceForensics/{compression}/{categories}/videos/{video}`,
    where `categorise` is `real`, `fake/Deepfakes`, `fake/FaceSwap`, `fake/Face2Face` or `fake/NeuralTextures`. `compression` is `c0`, `c23` or `c40`.
    The test videos we used in our experiments are given in `data/datasets/FaceForensics/test_list.txt`.
    * The directory structure of FakeAVCeleb: `your_dataset_root/FakeAVCeleb/videos/{categories}/{ethnic}/{gender}/{id}/{video}`,
      where `categories` includes `RealVideo-RealAudio`, `RealVideo-FakeAudio`, `FakeVideo-RealAudio` and `FakeVideo-FakeAudio`.
      For example, `your_dataset_root/FakeAVCeleb/videos/RealVideo-RealAudio/African/men/id00076/00109.mp4`.
    * The directory structure of KoDF: `your_dataset_root/KoDF/videos/{categories}/{id}/{video}`,
      where `categories` includes `original_videos`, `audio-driven`, `dffs`, `dfl` and `fo` (The videos we downloaded in `fsgan` do not contain audio,
      so we couldn't test them).
      The test videos we used in our experiments are given in `data/datasets/KoDF/test_list.txt`
3. Detect the faces and extract 68 face landmarks. Download the [RetinaFace](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) pretrained model,
   and put it to `checkpoints/Resnet50_Final.pth`. Run
   ```bash
   python preprocessing/face-alignment/landmark_extract.py --video_root $video_root --file_list $file_list --out_dir $out_dir
   ```
   - $video_root: root directory of videos.
   - $file_list: a txt file containing names of videos. We provide the filelists in the `data/datasets/` directory.
   - $out_dir: directory for saving landmarks.
4. To crop the mouth region from each video, run
   ```bash
   python preprocessing/align_mouth.py --video_root $video_root --file_list $file_list --landmark_dir $landmarks_dir --out_dir $out_dir
   ```
   - $out_dir: directory for saving cropped mouth videos.


## Evaluate

### Overview
The evaluation script tests the Audio-Visual Speech Representation model on three major face forgery datasets to measure its deepfake detection performance. The model evaluates synchronization between mouth movements (visual) and audio speech to distinguish real videos from deepfakes.

### Supported Datasets
- **FaceForensics++**: A comprehensive deepfake detection benchmark
- **FakeAVCeleb**: An audio-visual deepfake dataset  
- **KoDF**: Korean deepfake video dataset

### Performance
The model achieves the following AUC (Area Under the Curve) scores:

| FaceForensics++ | FakeAVCeleb | KoDF |
| :------------: | :-------------: | :-------------: |
| 97.6% | 99.0% | 91.7% |

### How to Run

1. Download the pretrained Audio-Visual Speech Representation model [here](https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt) and place it at `checkpoints/large_vox_iter5.pt`.

2. To evaluate on different datasets, run:
   ```bash
   python evaluate.py --video_root $video_root --file_list $file_list --mouth_dir $cropped_mouth_dir
   ```
   
   **Arguments:**
   - `--video_root`: Root directory containing the original video files
   - `--file_list`: Text file listing the names of videos to evaluate
   - `--mouth_dir`: Directory containing pre-processed cropped mouth regions extracted from videos
   - `--checkpoint_path`: Path to the AV-HuBERT checkpoint (default: `checkpoints/large_vox_iter5.pt`)
   - `--max_length`: Maximum video duration in seconds to process (default: 50)

## Single Video Inference
Use `single_infer.py` to compute the AV synchronization score for one video.

```bash
python single_infer.py \
   --video_root $video_root \
   --mouth_dir $cropped_mouth_dir \
   --video_path $video_path
```

Example:

```bash
python single_infer.py \
   --video_root data/datasets/FaceForensics/c23/Deepfakes/videos \
   --mouth_dir data/cropped_mouth/FaceForensics/c23/Deepfakes/videos \
   --video_path 000.mp4 \
   --checkpoint_path checkpoints/large_vox_iter5.pt \
   --max_seconds 50
```

Arguments:
- `--video_root`: Root directory containing the original videos.
- `--mouth_dir`: Root directory containing cropped mouth videos and corresponding `.wav` files.
- `--video_path`: Video path relative to `video_root` (or an absolute path under `video_root`).
- `--checkpoint_path`: Path to the AV-HuBERT checkpoint. Default is `checkpoints/large_vox_iter5.pt`.
- `--max_seconds`: Maximum video duration to process. Default is `50`.

Output:
- Prints the resolved mouth video path, audio path, and final AV sync score (`higher = better synchronization`).
