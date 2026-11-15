"""T2w scans from the paired low/high-field datasets, as one list of records.

Each record: dataset, subject, field_T, scanner (model and serial), session, path, and age / sex where the dataset
publishes them as participant data. Only image files and scanner fields are read; the LUMC 3 T sidecars still carry
patient identifiers, which are never opened here.

    ds006557   OpenNeuro, Vasa et al. 2025: GE SIGNA Premier 3 T CUBE T2w (1 mm) and axial T2w on two Hyperfine
               Swoop scanners (sessions HFC and HFE, days apart), 23 adults
    lumc       Zenodo 20281403 v5, van den Broek, Webb and Lena: axial T2w on one Hyperfine Swoop, 1-5 sessions per
               person (66 people), and for 11 of them a Philips Achieva 3 T T2w at matched resolution (acq-lowres).
               64 mT scans reconstructed finer than 1.2 mm in-plane (one, at 0.8 mm) are left out: interpolated
               pixels share their noise, so per-voxel noise measures do not apply
"""

import csv
import json
from pathlib import Path


def ds006557(root):
    root = Path(root)
    with open(root / "participants.tsv") as f:
        people = {r["participant_id"]: r for r in csv.DictReader(f, delimiter="\t")}
    out = []
    for sub in sorted(root.glob("sub-*")):
        info = people.get(sub.name.removeprefix("sub-"), {})
        person = {"age": float(info.get("age", "nan")), "sex": info.get("sex", "")}
        for ses, field, pattern in (("GE", 3.0, "_ses-GE_T2w"), ("HFC", 0.064, "_ses-HFC_acq-axi_T2w"),
                                    ("HFE", 0.064, "_ses-HFE_acq-axi_T2w")):
            p = sub / f"ses-{ses}" / "anat" / f"{sub.name}{pattern}.nii.gz"
            if not p.is_file():
                continue
            j = json.load(open(p.with_name(p.name.replace(".nii.gz", ".json"))))
            out.append({"dataset": "ds006557", "subject": sub.name, "field_T": field, "session": ses,
                        "scanner": f"{j.get('Manufacturer')} {j.get('DeviceSerialNumber')}", "path": p, **person})
    return out


def lumc(root):
    """root: the unzipped v5 folder's Data directory."""
    root = Path(root)
    tag = lambda j, t: (j.get(t) or {}).get("value") if isinstance(j.get(t), dict) else j.get(t)
    out = []
    for p in sorted((root / "64mT data").glob("sub-*/ses-*/anat/*_run-1_T2w.nii.gz")):
        j = json.load(open(p.with_name(p.name.replace(".nii.gz", ".json"))))
        serial = tag(j, "(0018, 1000)")
        spacing = tag(j, "(0028, 0030)")
        if spacing and float(str(spacing).strip("[]").split(",")[0]) < 1.2:
            continue                                 # reconstructed on a finer, interpolated grid (one scan)
        out.append({"dataset": "lumc", "subject": p.parts[-4], "field_T": 0.064, "session": p.parts[-3],
                    "scanner": f"Hyperfine {serial or 'LUMC (serial not recorded)'}", "path": p,
                    "age": float("nan"), "sex": ""})
    for p in sorted((root / "3T data").glob("sub-*/anat/*_acq-lowres_T2w.nii.gz")):
        out.append({"dataset": "lumc", "subject": p.parts[-3], "field_T": 3.0, "session": "3T",
                    "scanner": "Philips Achieva 3T", "path": p, "age": float("nan"), "sex": ""})
    return out


def rescan_pairs(scans):
    """Pairs of 64 mT scans of one person: ds006557's two scanners, and LUMC's first two sessions (one pair per
    person, so people with five sessions do not count five times)."""
    by = {}
    for s in scans:
        if s["field_T"] < 1:
            by.setdefault((s["dataset"], s["subject"]), []).append(s)
    return [tuple(sorted(v, key=lambda s: s["session"])[:2]) for v in by.values() if len(v) >= 2]
