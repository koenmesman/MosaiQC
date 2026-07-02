import random
from typing import Iterable, List, Sequence, Tuple


Run = List[int]  # [value, count]


def build_runs(arr: Sequence[int]) -> List[Run]:
    if not arr:
        return []

    runs: List[Run] = []
    current = arr[0]
    count = 1

    for value in arr[1:]:
        if value == current:
            count += 1
            continue
        runs.append([current, count])
        current = value
        count = 1

    runs.append([current, count])
    return runs


def _append_or_merge(runs: List[Run], value: int, count: int) -> None:
    if count <= 0:
        return
    if runs and runs[-1][0] == value:
        runs[-1][1] += count
    else:
        runs.append([value, count])


def merge_runs(runs: Iterable[Sequence[int]]) -> List[Run]:
    merged: List[Run] = []
    for value, count in runs:
        _append_or_merge(merged, int(value), int(count))
    return merged


def _remove_singletons_generic(runs: List[Run]) -> Tuple[List[Run], int]:
    removed = True
    singles = 0

    while True:
        found_singleton = False
        new_runs: List[Run] = []

        for value, count in runs:
            if count == 1 and not removed:
                singles += 1
                removed = True
                found_singleton = True
                continue
            new_runs.append([value, count])
            removed = False

        if not found_singleton:
            break

        runs = merge_runs(new_runs)

    return runs, singles


def remove_singletons_iteratively(arr: Sequence[int]) -> Tuple[List[Run], int]:
    runs = build_runs(arr)
    if not runs:
        return [], 0

    # Fast path for binary input: identical behavior to the previous iterative
    # implementation, but in linear time.
    if all(value in (0, 1) for value, _ in runs):
        singles = 0
        removed_prev = True
        removed_any = False
        merged_runs: List[Run] = []

        for value, count in runs:
            if count == 1 and not removed_prev:
                singles += 1
                removed_prev = True
                removed_any = True
                continue

            _append_or_merge(merged_runs, value, count)
            removed_prev = False

        # The old loop carries `removed_prev` into the next pass.
        # For binary runs, that can only trigger one extra removal: a leading singleton.
        if removed_any and (not removed_prev) and merged_runs and merged_runs[0][1] == 1:
            singles += 1
            merged_runs = merged_runs[1:]

        return merged_runs, singles

    return _remove_singletons_generic(runs)


def remove_duos(runs: Sequence[Sequence[int]]) -> Tuple[List[int], int]:
    if not runs:
        return [], 0

    singles = 0
    new_runs: List[Run] = []

    first_value, first_count = runs[0]
    _append_or_merge(new_runs, int(first_value), int(first_count))

    removed_prev = False

    # Keep the previous behavior where first and last runs are never removed.
    for value, count in runs[1:-1]:
        value = int(value)
        count = int(count)
        if count == 2 and not removed_prev:
            singles += 2
            removed_prev = True
            continue

        _append_or_merge(new_runs, value, count)
        removed_prev = False

    if len(runs) > 1:
        last_value, last_count = runs[-1]
        _append_or_merge(new_runs, int(last_value), int(last_count))

    result: List[int] = []
    for value, count in new_runs:
        result.extend([value] * count)
    return result, singles


if __name__ == "__main__":
    n = 10
    number = random.getrandbits(n)
    test_str = [int(i) for i in format(number, "b")]

    print(test_str)

    filtered, singles = remove_singletons_iteratively(test_str)
    print(filtered)

    new, s = remove_duos(filtered)
    print(s)

    singles += s
    print(singles)
    print(new)
