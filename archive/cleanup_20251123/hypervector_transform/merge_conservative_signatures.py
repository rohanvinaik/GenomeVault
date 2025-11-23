#!/usr/bin/env python3
"""
Merge signature sets using conservative union approach.

For each transform that appears in BOTH sets:
- Take the MORE CONSERVATIVE constraint (higher threshold)
- This ensures signatures generalize across datasets
"""

import json
from pathlib import Path
from collections import defaultdict


def merge_conservative_signatures(old_path: Path, new_path: Path, output_path: Path):
    """Merge two signature sets conservatively."""

    with open(old_path, 'r') as f:
        old_sigs = json.load(f)

    with open(new_path, 'r') as f:
        new_sigs = json.load(f)

    # Group by transform
    old_by_transform = defaultdict(list)
    for sig in old_sigs:
        old_by_transform[sig['transform']].append(sig)

    new_by_transform = defaultdict(list)
    for sig in new_sigs:
        new_by_transform[sig['transform']].append(sig)

    # Find transforms that appear in BOTH sets
    common_transforms = set(old_by_transform.keys()) & set(new_by_transform.keys())

    print(f"Old signatures: {len(old_sigs)}")
    print(f"New signatures: {len(new_sigs)}")
    print(f"Common transforms: {len(common_transforms)}")
    print()

    merged = []

    for transform in sorted(common_transforms):
        old_variants = old_by_transform[transform]
        new_variants = new_by_transform[transform]

        # For each constraint pattern, find matches and use more conservative
        for old_sig in old_variants:
            old_constraints = old_sig['constraints']

            for new_sig in new_variants:
                new_constraints = new_sig['constraints']

                # Check if they constrain the same lens(es)
                if set(old_constraints.keys()) == set(new_constraints.keys()):
                    # Use MORE CONSERVATIVE threshold (higher value)
                    conservative_constraints = {}
                    for lens in old_constraints.keys():
                        conservative_constraints[lens] = max(
                            old_constraints[lens],
                            new_constraints[lens]
                        )

                    # Estimate fixes as minimum of the two
                    min_fixes = min(old_sig['fixes'], new_sig['fixes'])

                    merged.append({
                        'transform': transform,
                        'constraints': conservative_constraints,
                        'fixes': min_fixes,
                        'breaks': 0,
                        'source': 'conservative_union'
                    })

                    print(f"{transform} {conservative_constraints}")
                    print(f"  OLD: {old_constraints} → {old_sig['fixes']} fixes")
                    print(f"  NEW: {new_constraints} → {new_sig['fixes']} fixes")
                    print(f"  MERGED (conservative): {conservative_constraints} → {min_fixes} fixes")
                    print()

    # Sort by fixes descending
    merged.sort(key=lambda x: -x['fixes'])

    print(f"Total merged signatures: {len(merged)}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"Saved to: {output_path}")

    return merged


if __name__ == '__main__':
    base_dir = Path('HDV_VALIDATION_PACKAGE/architecture_testing')

    for quant in ['float32', 'int8', 'int4', 'binary']:
        print("=" * 80)
        print(f"MERGING {quant.upper()} SIGNATURES")
        print("=" * 80)
        print()

        old_path = base_dir / 'aligned_10k' / 'exhaustive_search' / f'{quant}_exhaustive_search_results.json'
        new_path = base_dir / 'comparison_results' / 'retrained_signatures' / f'{quant}_exhaustive_search_results.json'
        output_path = base_dir / 'comparison_results' / 'conservative_union' / f'{quant}_exhaustive_search_results.json'

        merge_conservative_signatures(old_path, new_path, output_path)
        print()
        print()
