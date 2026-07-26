
import os, json, math, random
import numpy as np
import scipy.io
import datetime


CTD_FOLDER        = os.environ.get("L2R_CTD_DIR", "data/ctd")
EXCLUDE_KEYWORDS  = ('AUV_',)
OUTPUT_PATH       = os.path.join(CTD_FOLDER, "split_manifest.json")
SEED              = 42

RL_RATIO          = 0.30
LLM_TRAIN_RATIO   = 0.75
LLM_VAL_RATIO     = 0.15
# LLM_TEST_RATIO  = 1 - LLM_TRAIN_RATIO - LLM_VAL_RATIO = 0.10



def _datenum2datetime(dn):
    return (datetime.datetime.fromordinal(int(dn))
            + datetime.timedelta(days=float(dn) % 1)
            - datetime.timedelta(days=366))


def _dominant_season(time_dts):
    counts = [0, 0, 0, 0]  # spring summer fall winter
    for dt in time_dts:
        m = dt.month
        if m in (3, 4, 5):    counts[0] += 1
        elif m in (6, 7, 8):  counts[1] += 1
        elif m in (9, 10, 11):counts[2] += 1
        else:                  counts[3] += 1
    return int(np.argmax(counts))  # 0=spring 1=summer 2=fall 3=winter


def _season_dist(time_dts):
    counts = [0, 0, 0, 0]
    for dt in time_dts:
        m = dt.month
        if m in (3, 4, 5):    counts[0] += 1
        elif m in (6, 7, 8):  counts[1] += 1
        elif m in (9, 10, 11):counts[2] += 1
        else:                  counts[3] += 1
    return counts


SEASON_NAMES = ['spring', 'summer', 'fall', 'winter']



def _balanced_split(files, file_info, ratio, rng):
    
    n_select = max(1, round(len(files) * ratio))


    by_season = [[], [], [], []]
    for f in files:
        by_season[file_info[f]['dominant_season']].append(f)
    for g in by_season:
        rng.shuffle(g)


    selected = []
    season_iters = [iter(g) for g in by_season]
    while len(selected) < n_select:
        added = False
        for it in season_iters:
            if len(selected) >= n_select:
                break
            try:
                selected.append(next(it))
                added = True
            except StopIteration:
                pass
        if not added:
            break

    selected_set = set(selected)
    remaining = [f for f in files if f not in selected_set]
    return selected, remaining



def main():
    rng = random.Random(SEED)

    all_files = sorted(f for f in os.listdir(CTD_FOLDER) if f.endswith('.mat'))
    mat_files = [f for f in all_files
                 if not any(kw in f for kw in EXCLUDE_KEYWORDS)]

    file_info = {}
    for fname in mat_files:
        fpath = os.path.join(CTD_FOLDER, fname)
        try:
            data = scipy.io.loadmat(fpath)
        except Exception:
            continue
        if 'new_time_grid' not in data:
            continue
        tg = np.asarray(data['new_time_grid']).flatten()
        n  = int(len(tg))
        try:
            dts = [_datenum2datetime(dn) for dn in tg]
        except Exception:
            continue
        file_info[fname] = {
            'n_points':       n,
            'start':          str(dts[0].date()),
            'end':            str(dts[-1].date()),
            'dominant_season': _dominant_season(dts),
            'season_dist':    _season_dist(dts),
        }


    lt10 = [f for f, d in file_info.items() if d['n_points'] < 10]
    mid  = [f for f, d in file_info.items() if 10 <= d['n_points'] <= 32]
    gt32 = [f for f, d in file_info.items() if d['n_points'] > 32]


    rng.shuffle(gt32)
    rl_files,  llm_files  = _balanced_split(gt32, file_info, RL_RATIO, rng)


    llm_train, llm_rest   = _balanced_split(llm_files, file_info, LLM_TRAIN_RATIO, rng)
    llm_val,   llm_test   = _balanced_split(llm_rest,  file_info,
                                             LLM_VAL_RATIO / (1 - LLM_TRAIN_RATIO), rng)


    manifest: dict[str, str] = {}
    for f in lt10:      manifest[f] = 'test'
    for f in mid:       manifest[f] = 'validation'
    for f in rl_files:  manifest[f] = 'rl_train'
    for f in llm_train: manifest[f] = 'llm_train'
    for f in llm_val:   manifest[f] = 'llm_val'
    for f in llm_test:  manifest[f] = 'llm_test'

    result = {'manifest': manifest, 'file_info': file_info}
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)


    counts = {k: sum(1 for v in manifest.values() if v == k)
              for k in ('test', 'validation', 'rl_train', 'llm_train', 'llm_val', 'llm_test')}
    print(f"{'split':<12} {'文件数':>6}   季节分布(春夏秋冬)")
    print("-" * 50)
    for split_name in ('rl_train', 'llm_train', 'llm_val', 'llm_test', 'validation', 'test'):
        fnames = [f for f, v in manifest.items() if v == split_name]
        sd = [0, 0, 0, 0]
        for f in fnames:
            for i, c in enumerate(file_info[f]['season_dist']):
                sd[i] += c
        total = sum(sd) or 1
        pct = [f"{100*c/total:.0f}%" for c in sd]
        print(f"{split_name:<12} {len(fnames):>6}   {' '.join(pct)}")
    print(f"\n清单已写入 {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
