# Lightning AI Studio setup (for Spike work)

Connection info + step-by-step so a future session (or a future you) can
pick up where Spike A left off without re-deriving any of this.

## TL;DR for a fresh terminal

```bash
# 1. Put the CLI on PATH (it was pip-installed with --user)
export PATH="$HOME/Library/Python/3.11/bin:$PATH"

# 2. Confirm auth + Studio visibility
lightning list studios --teamspace danmarzari/default-project

# 3. SSH in (the config already lives in ~/.ssh/config)
ssh sole-sapphire-nqrv

# 4. Inside the Studio: activate the spike venv
cd /teamspace/studios/this_studio && source spike-venv/bin/activate
```

If the CLI auth gets wiped (or a new machine), the one-time setup is:
```bash
python3.11 -m pip install --user lightning-sdk
lightning login                       # browser-based OAuth
lightning configure ssh \
  --name sole-sapphire-nqrv \
  --teamspace danmarzari/default-project
```

## Account / Studio facts

| Field | Value |
|---|---|
| Teamspace | `danmarzari/default-project` |
| Primary Studio | `sole-sapphire-nqrv` |
| Studio mode | **CPU** (flip to GPU only for Spike D / per-voice fine-tune) |
| Free tier | 15 credits / month (~22 T4-hours) |
| Idle auto-stop | 4 hours (disk survives; click Start to resume) |
| Persistent disk | 358 GB at `/teamspace/studios/this_studio/` |
| Other Studio (stopped) | `scratch-studio-devbox` |

## Local Mac paths (already set up)

- **CLI binary**: `~/Library/Python/3.11/bin/lightning`
- **Credentials**: `~/.lightning/credentials.json` (⚠️ never commit — has `api_key`)
- **SSH private key**: `~/.ssh/lightning_rsa`
- **SSH public key**: `~/.ssh/lightning_rsa.pub` (auto-uploaded by `lightning configure ssh`)
- **SSH config entry**: `~/.ssh/config` → `Host sole-sapphire-nqrv`

## Filesystem layout on the Studio

```
/teamspace/studios/this_studio/          ← your persistent home; 358 GB free
├── Voice/                               ← rsync'd from Mac, phase-0-spikes branch
├── spike-venv/                          ← Python 3.12 venv with torch CPU + onnxruntime
├── spike-artifacts/
│   └── zipvoice_distill/                ← the 4 ONNX files cached from HF
│       ├── fm_decoder.onnx     (455 MB FP32)
│       ├── fm_decoder_int8.onnx (119 MB INT8)
│       ├── text_encoder.onnx    (17 MB FP32)
│       ├── text_encoder_int8.onnx (5 MB INT8)
│       ├── model.json
│       ├── tokens.txt
│       └── zipvoice_base.json
├── on_start.sh                          ← Lightning's per-start hook
├── on_stop.sh                           ← Lightning's per-stop hook
└── main.py                              ← template stub, unused
```

## Common operations from the Mac

### Push changed files up
```bash
# Use rsync over the SSH alias. Excludes cover the usual suspects.
rsync -az --exclude='.venv' --exclude='node_modules' --exclude='__pycache__' \
  --exclude='.DS_Store' --exclude='backend/data/profiles' \
  --exclude='backend/data/cache' --exclude='backend/data/config.json' \
  /Users/dmarzari/Server-Sites/Voice/ \
  sole-sapphire-nqrv:/teamspace/studios/this_studio/Voice/
```

### Pull results down
```bash
rsync -az sole-sapphire-nqrv:/teamspace/studios/this_studio/Voice/spikes/results/ \
  /Users/dmarzari/Server-Sites/Voice/spikes/results/
```

### Run a spike remotely (no interactive shell needed)
```bash
ssh sole-sapphire-nqrv \
  'cd /teamspace/studios/this_studio && source spike-venv/bin/activate && \
   cd Voice && python spikes/spike_a_lightning/spike_a.py'
```

### Studio lifecycle
```bash
# Check what's running (Studio credit burn status):
lightning list studios --teamspace danmarzari/default-project

# Start a stopped Studio:
lightning start studio --name sole-sapphire-nqrv --teamspace danmarzari/default-project

# Stop a running Studio (saves credits if it was on GPU; CPU mode is free):
lightning stop studio --name sole-sapphire-nqrv --teamspace danmarzari/default-project
```

**4-hour idle auto-stop:** on the free tier, the Studio auto-stops after
4 hours without activity. The persistent disk (`/teamspace/studios/this_studio/`,
including our `spike-venv/` and `spike-artifacts/`) is preserved. Just
run `lightning start studio --name ... --teamspace ...` or click Start
in the web UI to resume — everything is right where you left it.

## Gotchas we hit on first setup

- **`python3-venv` wasn't pre-installed** on the Studio. Needed
  `sudo apt-get install python3.12-venv` before `python3.12 -m venv` worked.
- **`rsync` wasn't pre-installed** on the Studio. Same fix:
  `sudo apt-get install rsync`.
- **`python3.11` doesn't exist** on the Studio — it's Ubuntu 24 with
  `python3.12` as the default. Our Voice Studio backend pins 3.11; the
  spike venv is 3.12. Don't mix requirements.txts between them.
- **CLI `--version` from credentials.json** — my existing
  `~/.lightning/credentials.json` still had a live `user_id` + `api_key`
  from some prior session, so `lightning login` was a no-op. If in a
  new machine and `lightning list studios` errors with "Teamspace-Owner
  None", run `lightning login` to refresh.
- **The `lightning` scripts aren't on default PATH** because pip-install
  with `--user` puts them in `~/Library/Python/3.11/bin/`. Either
  `export PATH` as above or symlink into `/usr/local/bin`.

## Security

- `~/.lightning/credentials.json` is already gitignored globally (it's
  outside the repo), but **never paste its contents anywhere**. The
  `api_key` is a live bearer token for your Lightning account.
- SSH keys at `~/.ssh/lightning_rsa{,.pub}` — private key is local-only.
  Lightning regenerates it if you run `lightning configure ssh --overwrite`.
- The Studio's persistent disk is private to your account but
  accessible to anyone with your SSH private key. Don't put secrets
  there (Voice Studio's `data/config.json` with a Reader bearer token,
  Drive keys, etc.) unless you specifically want them cloud-accessible.

## What lives where for Spike B and beyond

When you move on to Spike B (ORT-Web + WebGPU in Chrome), the ONNX
artifacts are already on the Studio's persistent disk. Options:

1. **Pull them to the Mac** for local Chrome testing:
   ```bash
   rsync -az sole-sapphire-nqrv:/teamspace/studios/this_studio/spike-artifacts/zipvoice_distill/ \
     /Users/dmarzari/Server-Sites/Voice/spikes/spike_b_ort_web_harness/
   ```
2. **Serve them from the Studio** (it has a public URL when you expose a port)
   and point your Mac's Chrome at that.
3. **Proxy through localhost** — run `ssh -L 8088:localhost:8088 sole-sapphire-nqrv`
   and start a `python -m http.server 8088` on the Studio.

Option 1 is simplest; option 3 is closest to how Reader will ultimately
consume them.
