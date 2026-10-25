# Credits and third-party licences

Blue Voice's own code is covered by `LICENSE`. These parts come from others and keep their own licences:

| What | Who | Licence |
|---|---|---|
| 250-word sign recognition model (`models/gs/`) | GestureSign by Shashank Yadav (see `core/gs/LICENSE_GestureSign.txt`) | MIT |
| Training data behind that model, and the replay clips in `models/gs/replay_clips.json` | Google, "Isolated Sign Language Recognition" (ASL Signs) | CC BY 4.0 |
| Hand, pose and face tracking | Google MediaPipe (models are downloaded on first run) | Apache 2.0 |
| Sentence-level signing research used by `tools/fsboard/` | FSboard, Georg et al. 2024 | CC BY 4.0 |

| Handwriting font "Great Vibes" (embedded in `web/index.html`) | The Great Vibes Pro Project Authors | SIL Open Font License 1.1 |
| Other welcome-screen fonts (loaded from Google Fonts when online) | Their authors | SIL Open Font License 1.1 |

## Attribution required by CC BY 4.0
- The 250-word model and replay clips derive from the Google *Isolated Sign Language Recognition* / ASL Signs dataset
  (https://www.kaggle.com/competitions/asl-signs), licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
  Changes: the data was converted to hand/pose landmark features, used to train the model by GestureSign's author, and a small
  set of clips was resampled for the replay preview. No endorsement by Google is implied.
- FSboard (Georg et al., 2024), CC BY 4.0, is referenced by the training scripts in `tools/fsboard/`. No FSboard data is included in this repository.

## Python packages
Blue Voice depends on OpenCV (Apache 2.0), MediaPipe (Apache 2.0), NumPy (BSD), aiohttp (Apache 2.0), ONNX Runtime (MIT)
and, for overlay mode, pywebview (BSD-3). They are installed from PyPI and are not copied into this repository.

Please keep this file when you copy or fork the project.
