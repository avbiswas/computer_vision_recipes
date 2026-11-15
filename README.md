# CV Speedrun

Four computer-vision projects of increasing difficulty, built for a YouTube video.

| Project | Folder | What it does |
|---|---|---|
| Meme search | `meme_search/` | CLIP baseline for searching memes by meaning |
| PDF parser | `pdf_parser/` | Layout detection (YOLO11) + small VLM -> structured markdown/JSON |
| Tennis tracking | `tennis/` | Ball / player / court tracking with a tiny distilled model (`tennis/lite/`) |
| World model | `world_model/lightroom/` | A playable "dream" of a 2D room: VQ-VAE tokenizer + Genie-style spatio-temporal transformer |

Datasets, model weights, training runs and rendered media are not in the repo; each project's scripts regenerate or download them.
`tennis/` expects [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) cloned into `tennis/third_party/`.

## World model quick start

```bash
cd world_model/lightroom
./record_all.sh && python pack.py      # record + pack episodes
./train_vq_ft.sh                       # tokenizer (see vqvae.py for the v2 losses)
./train_wm.sh                          # tokenize + world model
python play_dream.py --wm runs/dyn_k4/best.pt   # play: real room vs. dream
```
