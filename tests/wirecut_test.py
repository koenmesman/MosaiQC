test_str = [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
#test_str = [0, 0, 1, 1, 0, 0, 1, 1]
#test_str = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
test_str = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]

import random

n = 10
number = random.getrandbits(n)
test_str = [int(i) for i in format(number, '0b')]

print(test_str)

def build_runs(arr):
    if not arr:
        return []

    runs = []
    current = arr[0]
    count = 1

    for x in arr[1:]:
        if x == current:
            count += 1
        else:
            runs.append([current, count])
            current = x
            count = 1

    runs.append([current, count])
    return runs

def merge_runs(runs):
    if not runs:
        return []

    merged = [runs[0][:]]
    for value, count in runs[1:]:
        if merged[-1][0] == value:
            merged[-1][1] += count
        else:
            merged.append([value, count])

    return merged

def remove_singletons_iteratively(arr):
    runs = build_runs(arr)

    removed = True
    singles = 0

    while True:
        found_singleton = False
        new_runs = []
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

    result = []
    for value, count in runs:
        result.extend([value] * count)
    print(singles)
    return runs, singles

def remove_duos(runs):
    found_singleton = False
    new_runs = [runs[0]]
    singles = 0

    removed = False
    for value, count in runs[1:]:
        if count == 2 and not removed:
            removed = True
            singles += 2
            continue
        new_runs.append([value, count])
        removed = False
    print(new_runs)


    if runs[-1] != new_runs[-1]:
        new_runs.append(runs[-1])
        singles -= 2
    
    runs = merge_runs(new_runs)
    

    result = []
    for value, count in runs:
        result.extend([value] * count)
    return result, singles



filtered, singles = remove_singletons_iteratively(test_str)
print(filtered)
new, s = remove_duos(filtered)
print(s)
singles += s
print(singles)
print(new)