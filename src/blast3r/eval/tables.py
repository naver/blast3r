# Copyright (C) 2026-present Naver Corporation. All rights reserved.

"""Turn per-scene metrics into a CSV, a CLI table, and optionally LaTeX."""
import pandas as pd

# Column name -> label used in papers.
LABELS = {
    'Auc_30': 'AUC@30',
    'Racc_5': 'RRA@5', 'Racc_15': 'RRA@15', 'Racc_30': 'RRA@30',
    'Tacc_5': 'RTA@5', 'Tacc_15': 'RTA@15', 'Tacc_30': 'RTA@30',
    'ate_rmse': 'ATE RMSE',
}


def build_table(rows, index='scene'):
    """Per-scene rows plus an AVG row over the numeric columns."""
    df = pd.DataFrame(rows).set_index(index)
    avg = df.mean(numeric_only=True).to_frame().T
    avg.index = ['AVG']
    out = pd.concat([df, avg])
    out.index.name = index
    return out


def show(df, title, columns=None, float_format='%.2f'):
    shown = df if columns is None else df[[c for c in columns if c in df.columns]]
    shown = shown.rename(columns=LABELS)
    print(f'\n{title}')
    print(shown.to_markdown(floatfmt=float_format.lstrip('%')))


def write_csv(df, path):
    df.to_csv(path, float_format='%.6g')
    print(f'\nWrote {path}')


def to_latex(df, columns=None, float_format='%.1f'):
    shown = df if columns is None else df[[c for c in columns if c in df.columns]]
    shown = shown.rename(columns=LABELS)
    table = shown.to_latex(float_format=float_format)
    return table.replace('_', r'\_').replace('AVG', r'\midrule' '\n' 'AVG')
