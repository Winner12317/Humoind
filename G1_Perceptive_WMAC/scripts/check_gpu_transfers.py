"""Check that tensors keep their values when they move between this machine's GPUs.

Two-GPU training (``--sim-device``) moves observations, depth frames and actions
between cards on every control step. On some machines a *direct* GPU-to-GPU copy
(PCIe peer-to-peer) silently delivers wrong values, while a copy staged through
CPU memory is sound. This script measures both for every pair of cards, and
checks each card on its own, before any simulation or training starts.

    python scripts/check_gpu_transfers.py                 # every visible GPU
    python scripts/check_gpu_transfers.py --gpus 1 2      # one pair
    python scripts/check_gpu_transfers.py --trials 60 --json report.json

Exit status 0: at least one pair is usable for training (its host-route copies
and both cards are sound). The trainer uses the host route by default and
checksums every crossing, so a machine with broken direct copies still trains.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def system_report():
    """What usually decides whether direct GPU-to-GPU copies work."""
    report = {}
    try:
        report["kernel_command_line"] = Path("/proc/cmdline").read_text().strip()
    except OSError:
        report["kernel_command_line"] = None
    groups = Path("/sys/kernel/iommu_groups")
    try:
        report["iommu_groups"] = sum(1 for _ in groups.iterdir()) if groups.is_dir() else 0
    except OSError:
        report["iommu_groups"] = None
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,pci.bus_id,driver_version",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        report["nvidia_smi"] = query.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        report["nvidia_smi"] = None
    return report


def single_card(torch, device, trials, megabytes):
    """Wrong results out of ``trials`` for a host round trip and an on-card copy."""
    generator = torch.Generator().manual_seed(1)
    round_trip = on_card = 0
    for trial in range(trials):
        size = megabytes[trial % len(megabytes)] * 2**20
        original = torch.randint(0, 256, (size,), dtype=torch.uint8, generator=generator)
        on_device = original.to(device)
        round_trip += int(not torch.equal(on_device.cpu(), original))
        on_card += int(not torch.equal(on_device.clone().cpu(), original))
    return {"host_round_trip": [round_trip, trials], "on_card_copy": [on_card, trials]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", type=int, nargs="+", help="GPU indices to test (default: all visible)")
    parser.add_argument("--trials", type=int, default=30, help="copies per direction and route")
    parser.add_argument("--megabytes", type=int, nargs="+", default=[1, 64, 176],
                        help="copy sizes; 176 MB is what crosses per control step at 4,096 robots")
    parser.add_argument("--json", type=Path, help="also write the full report to this file")
    args = parser.parse_args(argv)
    import torch
    from src.wmac.device_placement import transfer_self_test
    if not torch.cuda.is_available():
        parser.error("No CUDA device is visible")
    gpus = args.gpus if args.gpus else list(range(torch.cuda.device_count()))
    if any(index < 0 or index >= torch.cuda.device_count() for index in gpus):
        parser.error(f"This process sees GPUs 0-{torch.cuda.device_count() - 1}")
    system = system_report()
    print("System")
    print(f"  kernel command line: {system['kernel_command_line']}")
    print(f"  IOMMU groups: {system['iommu_groups']} "
          f"({'IOMMU active' if system['iommu_groups'] else 'IOMMU off or not visible'})")
    for line in system["nvidia_smi"] or ():
        print(f"  GPU {line}")
    print(f"  PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

    cards = {index: single_card(torch, f"cuda:{index}", args.trials, args.megabytes) for index in gpus}
    print("\nEach card on its own (wrong/total)")
    for index, result in cards.items():
        print(f"  cuda:{index}: host round trip {result['host_round_trip'][0]}/{result['host_round_trip'][1]}, "
              f"on-card copy {result['on_card_copy'][0]}/{result['on_card_copy'][1]}")
    sound_cards = {index for index, result in cards.items() if not any(wrong for wrong, _ in result.values())}

    pairs, usable, direct_ok = {}, [], []
    if len(gpus) > 1:
        print("\nPairs (wrong/total): direct = one GPU-to-GPU copy, host = through CPU memory")
    for position, first in enumerate(gpus):
        for second in gpus[position + 1:]:
            a, b = f"cuda:{first}", f"cuda:{second}"
            report = transfer_self_test(a, b, trials=args.trials, megabytes=tuple(args.megabytes))
            peer = bool(torch.cuda.can_device_access_peer(first, second))
            pairs[f"{first},{second}"] = {"peer_access_reported": peer, **report}
            cells = "; ".join(f"{route}: " + ", ".join(f"{direction} {wrong}/{total}"
                              for direction, (wrong, total) in report[route].items()) for route in ("direct", "host"))
            print(f"  GPUs {first} and {second} (peer access {'yes' if peer else 'no'}): {cells}")
            host_sound = not any(wrong for wrong, _ in report["host"].values())
            if host_sound and {first, second} <= sound_cards:
                usable.append((first, second))
                if not any(wrong for wrong, _ in report["direct"].values()):
                    direct_ok.append((first, second))

    print("\nResult")
    broken_cards = sorted(set(gpus) - sound_cards)
    if broken_cards:
        print(f"  GPUs {broken_cards} return wrong values without any other card involved. "
              "That is a fault of the card, its slot, riser, power or driver state: do not train on them.")
    if len(gpus) < 2:
        print("  One GPU tested; pair tests need two.")
    elif usable:
        for first, second in usable:
            note = "direct copies also sound" if (first, second) in direct_ok else "direct copies WRONG, host route sound"
            print(f"  usable: --device cuda:{first} --sim-device cuda:{second}   ({note})")
        if len(direct_ok) < len(usable):
            print("  The trainer's default host route with checksums is safe on the pairs above. Wrong direct copies "
                  "usually mean the IOMMU (or PCIe ACS) is enabled: NVIDIA does not support peer-to-peer transfer "
                  "with the IOMMU on. Check `sudo dmesg | grep -i -e iommu -e IO_PAGE_FAULT -e xid` and "
                  "`sudo lspci -vvv | grep ACSCtl`.")
    else:
        print("  No pair is usable: copies are wrong even through CPU memory, or a card is faulty.")
    full = {"system": system, "cards": {str(k): v for k, v in cards.items()}, "pairs": pairs,
            "usable_pairs": usable, "pairs_with_sound_direct_copies": direct_ok,
            "trials": args.trials, "megabytes": args.megabytes}
    if args.json:
        args.json.write_text(json.dumps(full, indent=1) + "\n")
    return 0 if usable or (len(gpus) < 2 and not broken_cards) else 1


if __name__ == "__main__":
    raise SystemExit(main())
