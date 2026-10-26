"""Metrics and aggregation, kept separate from adaptation."""
import csv
import json
from pathlib import Path
import numpy as np

def classification_metrics(targets, predictions):
    y, p = np.asarray(targets).reshape(-1), np.asarray(predictions).reshape(-1)
    if len(y) != len(p) or not len(y):
        raise ValueError("Targets and predictions must have equal nonzero length.")
    return {"ACC": float(100*np.mean(y == p)),
            "BA": float(100*np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))}

def save_run(output, task, seed, indices, predictions, frozen, targets=None, **details):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{task}_seed{seed}"
    np.savez_compressed(out / (stem+'.npz'), indices=indices,
                        predictions=predictions, frozen=frozen)
    row = dict(task=task, seed=seed, n=len(indices), **details)
    if targets is not None:
        row.update(classification_metrics(targets, predictions))
        row.update({"zero_shot_"+k:v for k,v in classification_metrics(targets, frozen).items()})
    (out / (stem+'.json')).write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row), flush=True)
    return row

def summarize(folder):
    paths = sorted(Path(folder).glob('*_seed*.json'))
    rows = [json.loads(p.read_text()) for p in paths]
    if not rows or any('ACC' not in r for r in rows):
        raise ValueError("No complete scored runs found.")
    keys = [(r['task'], r['seed']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate task/order results.")
    tasks = sorted({r['task'] for r in rows})
    seeds = sorted({r['seed'] for r in rows})
    expected = {(t,s) for t in tasks for s in seeds}
    if set(keys) != expected:
        raise ValueError("Tasks must have the same set of stream orders.")
    metrics = ['ACC','BA','zero_shot_ACC','zero_shot_BA']
    per_task = []
    for task in tasks:
        rr = [r for r in rows if r['task']==task]
        if len({r['n'] for r in rr}) != 1:
            raise ValueError("Evaluation sample counts differ across orders.")
        if len({r.get('variant', '') for r in rr}) != 1:
            raise ValueError("Do not aggregate different method variants.")
        per_task.append(dict(task=task, n=rr[0]['n'], orders=len(seeds),
                             **{k:float(np.mean([r[k] for r in rr])) for k in metrics}))
    with (Path(folder)/'per_dataset.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(per_task[0]));w.writeheader();w.writerows(per_task)
    result = dict(tasks=len(tasks), seeds=seeds,
                  **{k:float(np.mean([r[k] for r in per_task])) for k in metrics})
    (Path(folder)/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return result
