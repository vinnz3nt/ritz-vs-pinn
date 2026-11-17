"""
One-off: move the old flat checkpoint folders into the study-centred layout.

    checkpoints/performance_study_<method>_<D>_<ts>/
        -> checkpoints/performance_study_<first ts>/<method>_D<D>/
    checkpoints/cont_v_discont_<method>_<problem>_<D>_<ts>/
        -> checkpoints/cont_v_discont_<first ts>/<method>_<problem>_D<D>/
    checkpoints/<D>D_<method>_sinus_sum_<ts>/
        -> checkpoints/scaling_study_<first ts>/<method>_D<D>/

Each old study becomes one study folder, named after the timestamp of its
earliest run. Folders are only moved (renamed), nothing is deleted or rewritten.
Anything else in checkpoints/ (e.g. discont_known_solution_* runs) is left alone.

Check the dry-run output first, and don't run --apply while a study is training.

Usage:
    python scripts/migrate_checkpoints.py           # dry run: print what would move
    python scripts/migrate_checkpoints.py --apply   # actually move
"""

import re
import sys
from collections import defaultdict

from ritz_vs_pinn.config import PATHS

TS = r"(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"

# study -> (pattern for old folder names, format for the new run name)
PATTERNS = {
    "performance_study": (
        re.compile(rf"performance_study_(?P<method>ritz|pinn)_(?P<D>\d+)_{TS}"),
        "{method}_D{D}",
    ),
    "cont_v_discont": (
        re.compile(rf"cont_v_discont_(?P<method>ritz|pinn)_(?P<problem>cont|discont)_(?P<D>\d+)_{TS}"),
        "{method}_{problem}_D{D}",
    ),
    "scaling_study": (
        re.compile(rf"(?P<D>\d+)D_(?P<method>ritz|pinn)_sinus_sum_{TS}"),
        "{method}_D{D}",
    ),
}


def plan():
    """{study: [(old_path, new_run_name, timestamp), ...]} for every old-style folder."""
    found = defaultdict(list)
    for path in sorted(PATHS.checkpoints.iterdir()):
        if not path.is_dir():
            continue
        for study, (pattern, run_fmt) in PATTERNS.items():
            m = pattern.fullmatch(path.name)
            if m:
                found[study].append((path, run_fmt.format(**m.groupdict()), m["ts"]))
                break
    return found


def main(apply):
    found = plan()
    if not found:
        print("Nothing to migrate.")
        return

    moves = []
    for study, runs in found.items():
        target = PATHS.checkpoints / f"{study}_{min(ts for _, _, ts in runs)}"
        names = [name for _, name, _ in runs]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            sys.exit(f"{study}: more than one old run for {sorted(dupes)}; "
                     "remove the ones you don't want first.")
        if target.exists():
            sys.exit(f"{target} already exists; not touching it.")
        print(f"{target.name}/  ({len(runs)} runs)")
        for old, name, _ in runs:
            print(f"    {old.name}  ->  {name}")
            moves.append((old, target / name))

    if not apply:
        print("\nDry run, nothing moved. Re-run with --apply to move.")
        return

    for old, new in moves:
        new.parent.mkdir(exist_ok=True)
        old.rename(new)
    print(f"\nMoved {len(moves)} folders.")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv[1:])
