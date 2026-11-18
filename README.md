# Computer Vision Recipes

From the YouTube channel **[Neural Breakdown with AVB](https://www.youtube.com/@avb_fj)**.

Four self-contained computer-vision projects to learn from. Each folder has its own code and instructions.

### Meme search — [`meme_search/`](meme_search/)
Describe a situation, get back a blank meme template with its text boxes pre-filled (CLIP fine-tuned on text-free memes).

![Meme search](images/meme.png)

### PDF parser — [`pdf_parser/`](pdf_parser/)
Detect page layout with a fine-tuned YOLO, then turn each region into structured markdown/JSON with a small VLM.

![PDF parser](images/pdf.png)

### Tennis tracking — [`tennis/`](tennis/)
Track the ball, players and court lines from broadcast video with a tiny distilled model.

https://github.com/user-attachments/assets/3b071be0-e0c0-4747-877c-ffd5e5f2cef6

### World model — [`world_model/lightroom/`](world_model/lightroom/)
A playable "dream" of a 2D room: a VQ-VAE tokenizer plus a Genie-style transformer that predicts the next frame from your actions.

https://github.com/user-attachments/assets/e05f140f-9753-4b4f-8452-d41ca88a3fd5

## Website: [cv.neuralavb.com](https://cv.neuralavb.com)

Every project with a page keeps it in its own `web/` folder: `index.html`, scripts, and a `site.json` that lists
the files to publish. `site/build.py` collects them into `dist/` and makes the landing page from the `site.json` files.
Everything runs in the browser: no API keys, no server code.

- **Text** (HTML, JS, small JSON) is in git. Cloudflare Workers Builds runs `python3 site/build.py` and
  `npx wrangler deploy` on every push to `main`, so page edits go live by themselves.
- **Binaries** (models, data, images, videos) are git-ignored. They live in the R2 bucket `cv-speedrun-assets` and
  pages load them from `/r2/<slug>/...`. Upload by hand after you change them; only new or changed files are sent:

```bash
python site/build.py --upload-r2           # needs `npx wrangler login`
python site/build.py --upload-r2 --local   # local preview: then `python site/build.py && npx wrangler dev`
```

To add a project: create `<project>/web/index.html` + `site.json` (format in the docstring of `site/build.py`),
upload its binaries, then push.

## Notes

Datasets, model weights, training runs and rendered media are not in the repo; each project's scripts regenerate or download them.
Tennis: `bash tennis/setup.sh`, then follow [`tennis/README.md`](tennis/README.md).

## Support

If you find this helpful, consider supporting on Patreon — it hosts all code, projects, slides, and write-ups from the YouTube channel.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)
