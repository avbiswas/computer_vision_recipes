# Computer Vision Recipes

Four self-contained computer-vision projects to learn from. Each folder has its own code and instructions.

| Project | Folder | What it does |
|---|---|---|
| Meme search | `meme_search/` | CLIP baseline for searching memes by meaning |
| PDF parser | `pdf_parser/` | Layout detection (YOLO11) + small VLM -> structured markdown/JSON |
| Tennis tracking | `tennis/` | Ball / player / court tracking with a tiny distilled model (`tennis/lite/`) |
| World model | `world_model/lightroom/` | A playable "dream" of a 2D room: VQ-VAE tokenizer + Genie-style spatio-temporal transformer |

Datasets, model weights, training runs and rendered media are not in the repo; each project's scripts regenerate or download them.
Tennis: `bash tennis/setup.sh`, then follow [`tennis/README.md`](tennis/README.md).

## World model quick start

```bash
cd world_model/lightroom
./record_all.sh && python pack.py      # record + pack episodes
./train_vq_ft.sh                       # tokenizer (see vqvae.py for the v2 losses)
./train_wm.sh                          # tokenize + world model
python play_dream.py --wm runs/dyn_k4/best.pt   # play: real room vs. dream
```
