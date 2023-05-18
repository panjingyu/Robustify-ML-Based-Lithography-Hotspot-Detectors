import os
import random


def main(test_dir, seed):
    png_dir = os.path.join(test_dir, 'png')
    png_files = os.listdir(png_dir)
    H_png_files = sorted(p for p in png_files if p.startswith('H'))
    N_png_files = sorted(p for p in png_files if p.startswith('N'))
    random.Random(seed).shuffle(H_png_files)
    random.Random(seed).shuffle(N_png_files)
    N =100
    for g in range(9):
        H_lines = [os.path.join(png_dir, p) for p in H_png_files[g*N:(g+1)*N]]
        N_lines = [os.path.join(png_dir, p) for p in N_png_files[g*N:(g+1)*N]]
        with open(f'config/test-num{N}-G{g+1}-seed{seed}.csv', 'w') as f:
            f.write('\n'.join(H_lines + N_lines))


if __name__ == '__main__':
    test_dir = 'data/vias-merge/test'
    seed =42
    main(test_dir, seed)
