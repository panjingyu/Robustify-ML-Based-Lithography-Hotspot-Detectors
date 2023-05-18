# Robustify ML-based Lithography Hotspot Detectors

Download data here: [Dropbox](https://www.dropbox.com/s/x6qpvw8r4lqefi0/data_vias-merge.zip?dl=1).

To train a model:

```bash
PYTHONPATH=$(pwd) python -u scripts/train.py --lr 0.01 --bs 128 --log ./log/$(date +%Y%m%d%H%M%S)
```

To test a model:

```bash
PYTHONPATH=$(pwd) python scripts/test.py --saved log/example --csv config/val-num100-seed42.csv -p 20
```

To attack a model:

```bash
PYTHONPATH=$(pwd) python -u scripts/attack.py --saved log/example -p 20 --csv val-num100-seed42
```