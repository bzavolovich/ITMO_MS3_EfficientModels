import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from equations import flops, memory, latency, energy

ROOT = Path(__file__).resolve().parent


def main():
    df = pd.read_csv(ROOT / 'results/measurements.csv')
    if len(df) != 132 or len(df[['S', 'B']].drop_duplicates()) != 132:
        raise ValueError('The complete grid must contain 132 distinct configurations.')
    if set(df.status) - {'ok', 'OOM'}:
        raise ValueError('Every configuration must have status ok or OOM.')
    for column in ['is_validation', 'predicted_oom']:
        df[column] = df[column].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
        if df[column].isna().any():
            raise ValueError(f'Invalid values in {column}.')
    successful = df[df.status == 'ok']
    if successful.empty or not np.isfinite(successful[['flops', 'memory', 'latency', 'energy']]).all().all():
        raise ValueError('Successful measurements must contain finite values.')
    if (successful[['flops', 'memory', 'latency', 'energy']] <= 0).any().any():
        raise ValueError('Successful measurements must contain positive values.')
    if successful.is_validation.all() or not successful.is_validation.any():
        raise ValueError('Successful calibration and validation configurations are required.')
    (ROOT / 'results/figures').mkdir(parents=True, exist_ok=True)
    batches = sorted(df.B.unique())
    valid = df[df.status == 'ok'].copy()
    calibration = valid[~valid.is_validation]
    s, b, measured = calibration.S.to_numpy(), calibration.B.to_numpy(), calibration.latency.to_numpy()

    def time_errors(log_theta):
        return np.log(latency(s, b, np.exp(log_theta))) - np.log(measured)

    fits = [least_squares(time_errors, np.log([measured.min()/2, c, w]),
                          bounds=(np.log([1e-8, 1e6, 1e5]), np.log([1, 1e15, 1e14])))
            for c in [1e12, 5e12] for w in [5e10, 3e11]]
    theta = np.exp(min(fits, key=lambda fit: fit.cost).x)
    power = np.exp(np.mean(np.log(calibration.energy.to_numpy() / latency(s, b, theta))))
    theta_energy = np.append(theta, power)
    (ROOT / 'results/theta.json').write_text(json.dumps({
        'latency': dict(zip(['tau_s', 'compute_flops_s', 'bandwidth_bytes_s'], theta.tolist())),
        'power_w': float(power)
    }, indent=2))
    print('tau [s], compute [FLOP/s], bandwidth [byte/s], power [W]:', theta_energy)

    predictions = {
        'flops': (flops, 1e9, 'GFLOP (count from layer shapes)'),
        'memory': (memory, 2**30, 'GiB'),
        'latency': (lambda s, b: latency(s, b, theta), .001, 'ms'),
        'energy': (lambda s, b: energy(s, b, theta_energy), 1, 'J')
    }
    for name, (predict, scale, unit) in predictions.items():
        fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True)
        for ax, B in zip(axes.flat, batches):
            ss = np.arange(32, 513, 16)
            ax.plot(ss, predict(ss, B)/scale, 'k-', label='Prediction')
            for held, marker, label in [(False, 'o', 'Calibration'), (True, 'x', 'Validation')]:
                part = valid[(valid.B == B) & (valid.is_validation == held)]
                ax.scatter(part.S, part[name]/scale, marker=marker, label=label)
            ax.set(title=f'B={B}', xlabel='S [pixels]', ylabel=unit, yscale='log')
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(ROOT / f'results/figures/{name}.png', dpi=150)
        plt.close(fig)
        test = valid[valid.is_validation]
        error = np.abs(predict(test.S.to_numpy(), test.B.to_numpy()) / test[name].to_numpy() - 1)
        print(name, 'validation median absolute error [%]:', round(100*np.median(error), 2))
    print(pd.crosstab(df.status, df.predicted_oom, colnames=['Predicted OOM']))

    print('Configurations:', len(df))
    print('OOM:', int((df.status == 'OOM').sum()))
    print('Calibration:', len(calibration), 'Validation:', int(valid.is_validation.sum()))
    print('Maximum measured memory [GiB]:', float(valid.memory.max() / 2**30))
    print('Predicted memory at S=512, B=256 [GiB]:', float(memory(512, 256) / 2**30))


if __name__ == '__main__':
    main()
