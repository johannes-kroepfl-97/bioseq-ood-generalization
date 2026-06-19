# RunPod runbook — gfp sanity run

Goal: run `MODE="gfp_sanity"` (full gfp, 5 encoder / 10 method trials, top-2, 2 seeds)
on a single **RTX 4090**, inspect the outputs, then scale up later for the poster.

Estimated: ~212 training runs, ~11–18 h, ~$7–12.

---

## 0. On your laptop — pack the project (once)

From the folder that *contains* the repo (`C:\Users\kroep\Desktop\AI`), in PowerShell:

```powershell
cd C:\Users\kroep\Desktop\AI
tar -czf bioseq.tar.gz `
  --exclude=".venv" --exclude=".git" --exclude="results" --exclude="results_phases" `
  --exclude="_pretest_backup*" --exclude="__pycache__" `
  bioseq-ood-generalization
```

This bundles the code, `config/`, `data/gfp/`, **and** `uv.lock` + `.python-version`
(so the pod reproduces your exact environment). `results/` is excluded — those are
regenerated on the pod.

---

## 1. Launch the pod

On runpod.io → Deploy:
- **GPU:** RTX 4090
- **Template:** "RunPod PyTorch" (CUDA preinstalled)
- **Volume:** add a ~20 GB **Network Volume** mounted at `/workspace` (persists if the pod restarts)
- Start it, then open the **web terminal** (or SSH / JupyterLab).

---

## 2. Get the project onto the pod

Upload `bioseq.tar.gz` to `/workspace` (drag-drop in JupyterLab, or `runpodctl`/`scp`), then:

```bash
cd /workspace
tar -xzf bioseq.tar.gz
cd bioseq-ood-generalization
rm -rf config/tuned results_phases   # drop any DEBUG-run artifacts so Phase B trains fresh on full data
```

> Important: `config/tuned/` and `results_phases/` may contain tutorial/debug outputs.
> Deleting them forces a clean full-data run. (If they weren't in the tarball, this is a no-op.)

---

## 3. Build the environment with uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux installer (the pod is Linux)
source $HOME/.local/bin/env
uv sync --extra dev                               # installs torch (cu121) + everything, from uv.lock
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # must print True
```

---

## 4. Timing check — your FIRST real run is the probe

Just start the run (next step) and look at the log: the time between the first
`=== Phase B | gfp | cnn ...` line and its first `trial 0: val_id_mae = ...` line is
**one full-data ERM run**. Multiply by ~212 for a total estimate.
- If a run takes ~3–5 min → on track (~11–18 h total). Continue.
- If a run takes >15 min → something's off (e.g., CPU-bound); stop with `Ctrl-C` and tell me.

Because the run is **resumable and cheap**, "committing" here is low-risk — you can stop
any time and the finished runs are already saved.

---

## 5. Launch (detached, so it survives disconnects)

```bash
tmux new -s run
uv run python pipeline_phases.py 2>&1 | tee run.log
# detach with: Ctrl-b then d        # reattach later with: tmux attach -t run
```

`MODE` is already set to `gfp_sanity` in `pipeline_phases.py`, so no flags needed.

---

## 6. Watch it live (from a second terminal / tab)

```bash
tail -f run.log                                  # phase/trial/protocol + MAE per run
watch -n 10 'wc -l results_phases/*.csv'         # Phase F rows accumulating live
pip install nvitop && nvitop                     # GPU utilization / memory
```

Phase F writes one row per run as it finishes (incremental), so `phaseF_all_protocols.csv`
and `phaseF_by_mut_dist.csv` grow in real time.

---

## 7. When it finishes — inspect (sanity checklist)

```bash
ls -la results_phases/        # expect phaseB..G + phaseB_all_trials + phaseF/G_by_mut_dist
grep -c "Stage-1 reused" run.log     # pseudo-labeling loaded the baseline (should be > 0)
grep -i "NON-FINITE\|Traceback" run.log   # should be empty
```
Then eyeball: `phaseG_analysis.csv` (lift_close/lift_far > 0?), `phaseG_by_mut_dist.csv`
(curves degrade with distance? `report_mae_std` populated?), `phaseB_all_trials.csv`
(best trials inside the search ranges, not at a boundary?).

---

## 8. Download results to your laptop

```bash
# from your laptop (or use JupyterLab download / runpodctl)
scp -r <pod>:/workspace/bioseq-ood-generalization/results_phases ./results_phases_gfp
scp -r <pod>:/workspace/bioseq-ood-generalization/config/tuned   ./tuned_gfp
```

---

## 9. Stop the pod (stop billing)

Stop or terminate the pod in the dashboard. With the Network Volume, your data + results
remain and you can re-attach for the aav/tfbind8 or the larger poster run later.

---

### If it gets interrupted
Just relaunch step 5. Phases B–E resume from their committed files; **Phase F resumes
run-by-run** (it skips the cells already in `phaseF_all_protocols.csv`). Nothing is lost.
