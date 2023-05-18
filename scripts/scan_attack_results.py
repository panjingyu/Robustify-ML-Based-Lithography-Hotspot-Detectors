import os
import re
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import tqdm


VAL_NUM = 100


class UnfinishedAttackException(BaseException):
    pass


def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--log', type=str, default='log-drc.norm')
    return parser.parse_args()


def parse_attack_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    if len(lines) < VAL_NUM or not lines[-1].endswith('INFO:Finished\n'):
        raise UnfinishedAttackException
    ptr = -2
    while True:
        found = re.findall(r'(\d+)\s+/\s+(\d+)', lines[ptr])
        if len(found) == 0:
            ptr -= 1
        else:
            success, trials = found[0]
            break
    return int(success), int(trials)


def parse_tb(tb_path, scalars):
    ea = event_accumulator.EventAccumulator(
        tb_path,
        size_guidance={event_accumulator.SCALARS: 400},
    )
    # make sure the scalars are in the event accumulator tags
    ea.Reload()
    assert all(
        s in ea.Tags()['scalars'] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


def parse_exp_title(exp: str):
    exp = exp.split('@')[0]
    exp_cure = exp[exp.find('cure'):]
    l_idx = exp_cure.find('L')
    h_idx = exp_cure.find('H')
    cure_l = exp_cure[l_idx+1:h_idx]
    cure_h = exp_cure[h_idx+1:]
    return float(cure_l), float(cure_h)


def parse_attack_hp(log_title):
    secs = log_title.split('.')
    p = list(filter(lambda x: x.startswith('p'), secs))
    if len(p) == 0 and log_title == 'attack.log':
        return 20
    assert len(p) == 1, log_title
    p = p[0][1:]
    return int(p)


def parse_all(attack_logs, exp_dir, exp):
    cure_l, cure_h = parse_exp_title(exp)
    # parse tensorboard file
    tb_files = list(x for x in os.listdir(exp_dir) if x.startswith('event'))
    tb_path = os.path.join(exp_dir, tb_files[0])
    assert len(tb_files) == 1
    tb_out = parse_tb(tb_path, ['acc/val-AUC', 'acc/val-H', 'acc/val-N'])
    parsed = {
        'dir': exp_dir,
        'val-AUC': tb_out['acc/val-AUC'].iloc[-1].value,
        'val-H': tb_out['acc/val-H'].iloc[-1].value,
        'val-N': tb_out['acc/val-N'].iloc[-1].value,
        'cure-L': cure_l,
        'cure-H': cure_h,
    }
    for attack_log in attack_logs:
        max_perts = parse_attack_hp(attack_log)
        attack_log = os.path.join(exp_dir, attack_log)
        success, trials = parse_attack_log(attack_log)
        parsed.update({
            f'success (p={max_perts})': success,
            f'trials (p={max_perts})': trials,
        })
    return parsed


def main(log_dir):
    exp_list = sorted(os.listdir(log_dir))
    parsed_exp = dict()
    for exp in tqdm.tqdm(exp_list):
        exp_dir = os.path.join(log_dir, exp)
        assert os.path.isdir(exp_dir)
        attacks = sorted(x for x in os.listdir(exp_dir)
                         if x.startswith('attack') and x.endswith('.log'))
        if len(attacks) == 0:
            continue    # no attacks found, skip
        try:
            parsed_exp[exp] = parse_all(attacks, exp_dir, exp)
        except UnfinishedAttackException:
            continue
    keys = sorted(parsed_exp.keys())
    # attrs = list(parsed_exp[keys[0]].keys())
    attrs = set()
    for exp in parsed_exp:
        attrs.update(set(parsed_exp[exp].keys()))
    attrs = list(attrs)

    d = {a: dict() for a in attrs}
    for a in attrs:
        for exp in parsed_exp:
            if a in parsed_exp[exp]:
                d[a].update({exp: parsed_exp[exp][a]})

    # df = pd.DataFrame({a: [parsed_exp[k][a] for k in keys] for a in attrs})
    df = pd.DataFrame(d)
    df.to_excel(f'stats/{os.path.basename(log_dir)}.xlsx')


if __name__ == '__main__':
    args = parse_args()
    main(args.log)
