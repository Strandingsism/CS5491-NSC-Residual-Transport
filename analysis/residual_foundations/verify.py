"""Small mathematical checks; not a Transformer benchmark or NSC reproduction.

Requires numpy and matplotlib. Run this file; outputs are written beside it.
Spectral sums below use realized SVD values, not the MP specification scorer.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent


def psi(w):
    return float(np.log1p(np.linalg.svd(w, compute_uv=False) ** 2).sum())


def zero_branch_check():
    d, eta = 4, 0.1
    eye = np.eye(d)
    results = []
    for name, w1 in [("zero_output_only", eye.copy()), ("both_matrices_zero", np.zeros_like(eye))]:
        w2 = np.zeros_like(eye)
        x, target = eye, 2 * eye
        h = w1 @ x
        f = x + w2 @ h
        error = f - target
        # L = 0.5 * ||f-target||_F^2, without averaging.
        grad2 = error @ h.T
        grad1 = w2.T @ error @ x.T
        new1, new2 = w1 - eta * grad1, w2 - eta * grad2
        next_f = x + new2 @ new1 @ x
        # Jacobian is stacked in column-major output order.
        jtheta = np.concatenate([np.kron(x.T, w2), np.kron(h.T, eye)], axis=1)
        # Central finite differences verify both parameter blocks.
        params = np.concatenate([w1.ravel(order="F"), w2.ravel(order="F")])
        def evaluate(p):
            a = p[:d*d].reshape((d,d), order="F")
            b = p[d*d:].reshape((d,d), order="F")
            return (x + b @ a @ x).ravel(order="F")
        numerical = np.empty_like(jtheta)
        for i in range(len(params)):
            direction = np.zeros_like(params)
            direction[i] = 1e-6
            numerical[:, i] = (evaluate(params+direction)-evaluate(params-direction))/(2e-6)
        fd_error = float(np.max(np.abs(numerical-jtheta)))
        assert fd_error < 1e-8
        assert np.allclose(eye+w2@w1, eye)
        results.append(dict(case=name, forward_branch_energy=0.0,
            input_jacobian_condition=1.0, tangent_rank=int(np.linalg.matrix_rank(jtheta)),
            loss_before=float(0.5*np.sum(error**2)),
            loss_after=float(0.5*np.sum((next_f-target)**2)),
            parameter_jacobian_fd_error=fd_error))
    assert np.isclose(results[0]["loss_after"], 1.62)
    assert np.isclose(results[1]["loss_after"], 2.0)
    return results


def coordinate_check():
    # f=a*theta, at theta=1/a: same input-output map f(x)=x.
    rows=[]
    for a, eta in [(1.0,0.1),(10.0,0.1),(10.0,0.001)]:
        theta=1.0/a
        error=a*theta-2.0
        theta_new=theta-eta*a*error
        rows.append(dict(scale=a, learning_rate=eta, initial_output=a*theta,
            tangent_kernel=a*a, loss_after=0.5*(a*theta_new-2.0)**2))
    assert np.isclose(rows[0]["loss_after"], rows[2]["loss_after"])
    return rows


def depth_checks():
    d, seeds = 32, 24
    rows=[]
    for depth in [4,16,64,128]:
        # Paired initializations across integrated strengths.
        for seed in range(seeds):
            rng=np.random.default_rng(seed)
            weights=rng.normal(size=(depth,d,d))/np.sqrt(d)
            raw=sum(psi(w) for w in weights)/d
            for t in [0.1,1.0,4.0]:
                alpha=np.sqrt(t/depth)
                jac=np.eye(d)
                for w in weights:
                    jac=(np.eye(d)+alpha*w)@jac
                sv=np.linalg.svd(jac,compute_uv=False)
                rows.append(dict(depth=depth,seed=seed,strength=t,
                    raw_matrix_spectral_sum_per_dim=raw,
                    scaled_matrix_spectral_sum_per_dim=sum(psi(alpha*w) for w in weights)/d,
                    input_jacobian_log_condition=float(np.log(sv[0]/sv[-1])),
                    input_jacobian_energy=float(np.sum(sv**2)/d)))
    with (OUT/'depth_runs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summary=[]
    for depth in [4,16,64,128]:
        for t in [0.1,1.0,4.0]:
            selected=[r for r in rows if r['depth']==depth and r['strength']==t]
            entry=dict(depth=depth,strength=t,seeds=seeds)
            for key in list(rows[0])[3:]:
                a=np.array([r[key] for r in selected])
                entry[key]=dict(mean=float(a.mean()),std=float(a.std(ddof=1)))
            summary.append(entry)
    return summary


def plot(summary,zero):
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    subset=[r for r in summary if r['strength']==1.0]
    x=[r['depth'] for r in subset]
    for key,label,color in [('raw_matrix_spectral_sum_per_dim','Sum psi(W) / d','#ad4934'),
        ('scaled_matrix_spectral_sum_per_dim','Sum psi(alpha W) / d','#236e96')]:
        axes[0].plot(x,[r[key]['mean'] for r in subset],'-o',label=label,color=color)
    axes[0].set(xlabel='Depth L',ylabel='Realized matrix score / d',yscale='log',
        title='Same integrated strength: alpha = 1/sqrt(L)')
    axes[0].legend(fontsize=8)
    for t,color in [(0.1,'#236e96'),(1.0,'#4f865a'),(4.0,'#ad4934')]:
        rs=[r for r in summary if r['strength']==t]
        key='input_jacobian_log_condition'
        axes[1].errorbar(x,[r[key]['mean'] for r in rs],
            yerr=[r[key]['std'] for r in rs],fmt='-o',capsize=3,
            label=f't = {t}',color=color)
    axes[1].set(xlabel='Depth L',ylabel='ln(condition number of J_x)',
        title='Propagation depends on integrated strength')
    axes[1].legend(fontsize=8)
    axes[2].bar(['W1 = I, W2 = 0','W1 = W2 = 0'],[r['loss_after'] for r in zero],
        color=['#236e96','#ad4934'])
    axes[2].axhline(2,color='gray',linestyle='--',label='Initial loss = 2')
    axes[2].set(ylabel='Loss after one SGD step',ylim=(0,2.4),
        title='Both: zero branch output and J_x = I')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
    fig.savefig(OUT/'checks.png',dpi=180)
    fig.savefig(OUT/'checks.pdf')


if __name__=='__main__':
    zero=zero_branch_check()
    coords=coordinate_check()
    depth=depth_checks()
    same_logdet=dict(identity=psi(np.eye(2)),rank_one=psi(np.diag([np.sqrt(3),0])))
    assert np.isclose(same_logdet['identity'],same_logdet['rank_one'])
    results=dict(zero_branch=zero,coordinate_scaling=coords,equal_logdet=same_logdet,
        depth_summary=depth,notes='Synthetic linear residual models only; SVD scores are not official NSC-MP.')
    (OUT/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    plot(depth,zero)
    print(json.dumps(dict(zero_branch=zero,coordinate_scaling=coords,equal_logdet=same_logdet),indent=2))
    for r in depth:
        if r['strength']==1.0 or r['depth']==64:
            print(r['depth'],r['strength'],*[round(r[k]['mean'],4) for k in
                ['raw_matrix_spectral_sum_per_dim','scaled_matrix_spectral_sum_per_dim',
                 'input_jacobian_log_condition','input_jacobian_energy']])
