# Replay website

`replay/` is a second website, separate from the control app (`python main.py app`).
It plays back flights that were recorded beforehand, so it needs no Python, no
PyBullet and no server of its own: any static host can serve it for free and it
never goes to sleep.

What it shows: eight quadcopters crossing a city past one defence platform, flown by
each model on the same starts and goals, with the outcome of every drone, a timeline
you can scrub, playback speed, a follow camera and a table that compares all models
in the chosen city.

| | Control app (`main.py app`) | Replay site (`replay/`) |
|---|---|---|
| Physics | live, on your PC | none, recorded flights |
| You can | place drones, goals and defences, pilot, train | watch, scrub, compare models |
| Needs | Python, PyBullet, a running server | any static web host |

## 1. Record the flights

```bash
python scripts/record_replays.py --episodes 5
```

This flies every model in every city with the paper's evaluation settings
(`configs/gen_experiment.yaml`, collision reflex on) and writes `replay/data/`
(about 60 MB). It takes 10 to 20 minutes. Models and cities are listed at the top
of the script; `--models best planner --cities delhi_cp` records a subset.

`replay/data/` is in `.gitignore`: the recordings are build output and are uploaded
to the host, not committed to this repository.

## 2. Try it locally

```bash
cd replay
python -m http.server 8800
```

Open <http://localhost:8800>. (Opening `index.html` straight from disk does not work:
browsers block `fetch` on `file://` pages.)

## 3. Put it online (free)

Pick one. All three serve the folder as it is.

### Option A: Hugging Face Space (Static)

Static Spaces are free on every account and stay online.

1. On huggingface.co, **New Space** → name `swarmsim-replay` → SDK **Static** →
   template **Blank** → **Public** → **Create Space**.
2. Upload from your PC:

   ```bash
   pip install -U huggingface_hub
   hf auth login                     # opens the browser, or paste a write token from Settings → Access Tokens
   hf upload <your-hf-name>/swarmsim-replay replay . --repo-type space
   ```

   (`huggingface-cli upload ...` with the same arguments works on older versions.)
3. The site is at `https://huggingface.co/spaces/<your-hf-name>/swarmsim-replay`
   and, without the Hugging Face frame, at `https://<your-hf-name>-swarmsim-replay.static.hf.space`.

Keep the `README.md` that the Space created: its header (`sdk: static`) is what
makes it a static site. The upload above does not overwrite it because `replay/`
has no README of its own.

### Option B: GitHub Pages

Use a separate public repository for the site, in a folder outside this one, so the
recordings do not bloat the code repository. In PowerShell, from the `swarmsim` folder:

```powershell
# once: create an empty public repo named swarmsim-replay on github.com, then
Copy-Item -Recurse replay ..\swarmsim-replay-site
cd ..\swarmsim-replay-site
New-Item .nojekyll -ItemType File      # serve the files as they are
git init -b main
git add .
git commit -m "Replay site"
git remote add origin https://github.com/<your-github-name>/swarmsim-replay.git
git push -u origin main
```

On github.com: the repo → **Settings** → **Pages** → Source **Deploy from a branch**,
branch `main`, folder `/ (root)` → **Save**. After a minute or two the site is at
`https://<your-github-name>.github.io/swarmsim-replay/`.

To update it later: record again, copy `replay\*` over `..\swarmsim-replay-site`
(`Copy-Item -Recurse -Force replay\* ..\swarmsim-replay-site`), then in that folder
`git add -A; git commit -m "New recordings"; git push`.

### Option C: Firebase Hosting

```bash
npm install -g firebase-tools
firebase login
firebase init hosting               # public directory: replay   single-page app: No
firebase deploy --only hosting
```

The free plan has a small daily download allowance, and every visitor downloads
several megabytes of city points, so A or B suits a link you share widely.

## Sharing a particular flight

The address bar always holds the current choice, for example
`.../#city=delhi_cp&model=best&ep=2`. Copy it to send someone straight to that flight.

## Keys

Space play/pause · ← → back/forward 5 s (Shift: 1 s) · R restart · N next episode ·
1 to 8 pick a drone · F follow it · H or Home reset the view.
Mouse: drag to turn, right-drag to slide, wheel to zoom. Touch: one finger turns,
two fingers slide and pinch to zoom.
