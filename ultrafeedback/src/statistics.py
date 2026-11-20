"""Proposal H2 GEE and H3 clustered summaries. Estimates require real completed runs."""
import math
from .experiment import clustered_interval, holm


def h2_gee(comparisons, configs):
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from patsy import build_design_matrices
    from scipy.stats import norm
    by_id = {c['config_id']: c for c in configs}
    rows = []
    for r in comparisons:
        c = by_id[r['config_id']]
        if c.get('sampling_control') or r.get('prompt_variant') != 'primary':
            continue
        rows.append(dict(r, thinking=int(c['thinking']), log2_size=math.log2(float(c['checkpoint'].split('-')[-1][:-1]))))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No completed primary comparisons')
    output, hypotheses = {}, []
    for bias, part in df.groupby('bias'):
        if part.prompt_id.nunique() < 20 or part.log2_size.nunique() < 3 or part.thinking.nunique() < 2:
            output[bias] = {'status': 'insufficient_clusters_or_configurations'}
            continue
        fit = smf.gee('flip ~ edited * log2_size * thinking', groups='prompt_id', data=part,
                      weights=part.weight, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Independence()).fit(cov_type='robust')
        beta = np.asarray(fit.params)
        covariance = np.asarray(fit.cov_params())
        if not fit.converged or not np.isfinite(beta).all() or not np.isfinite(covariance).all():
            output[bias] = {'status': 'fit_not_usable'}
            continue
        base = part[['pair_id', 'config_id', 'log2_size', 'thinking']].drop_duplicates().copy()

        def excess_design(frame, edited):
            frame = frame.assign(edited=edited)
            return np.asarray(build_design_matrices([fit.model.data.design_info], frame)[0])

        def expected_excess(frame, coefficients):
            from scipy.special import expit
            # Explicit row sums avoid problematic BLAS vector products in some Mac NumPy builds.
            return expit(np.sum(excess_design(frame, 1) * coefficients, axis=1)) - expit(np.sum(excess_design(frame, 0) * coefficients, axis=1))

        def effect(coefficients, kind):
            if kind == 'thinking':
                return float(np.mean(expected_excess(base.assign(thinking=1), coefficients) -
                                     expected_excess(base.assign(thinking=0), coefficients)))
            # Avoid extrapolating a doubling beyond the largest planned 8B model.
            eligible = base[base.log2_size <= math.log2(4)]
            return float(np.mean(expected_excess(eligible.assign(log2_size=eligible.log2_size + 1), coefficients) -
                                 expected_excess(eligible, coefficients)))

        results = {}
        for kind in ['size_per_doubling', 'thinking']:
            estimate = effect(beta, kind)
            eps = 1e-5
            grad = np.array([(effect(beta + np.eye(len(beta))[i] * eps, kind) -
                              effect(beta - np.eye(len(beta))[i] * eps, kind)) / (2 * eps) for i in range(len(beta))])
            variance = float(np.sum(grad[:, None] * covariance * grad[None, :]))
            if not np.isfinite(estimate) or not np.isfinite(grad).all() or not np.isfinite(variance):
                results[kind] = {'status': 'nonfinite_marginal_effect'}
                continue
            se = float(np.sqrt(max(0, variance)))
            if se <= 0:
                results[kind] = {'status': 'zero_or_invalid_standard_error'}
                continue
            p_pos = float(norm.sf((estimate - .03) / se))
            p_neg = float(norm.cdf((estimate + .03) / se))
            p_equiv = float(max(norm.sf((estimate + .03) / se), norm.cdf((estimate - .03) / se)))
            results[kind] = {'average_marginal_effect_points': 100 * estimate,
                             'robust_se_points': 100 * se,
                             'ci95_points': [100 * (estimate - 1.96 * se), 100 * (estimate + 1.96 * se)]}
            for label, p in [('greater_than_plus_3_points', p_pos), ('less_than_minus_3_points', p_neg), ('equivalent_within_3_points', p_equiv)]:
                hypotheses.append({'bias': bias, 'effect': kind, 'hypothesis': label, 'p': p})
        output[bias] = {'status': 'fit', 'clusters': int(part.prompt_id.nunique()),
                        'pairs': int(part.pair_id.nunique()), 'effects': results,
                        'coefficients': {k: float(v) for k, v in fit.params.items()}}
    adjusted = holm([h['p'] for h in hypotheses])
    for h, p in zip(hypotheses, adjusted):
        h['holm_p'] = p
    return {'models': output, 'hypotheses': hypotheses,
            'family': 'H2: both biases × size/thinking × positive/negative/equivalence tests; freeze before main run',
            'method': 'Binomial GEE, independence working correlation, prompt-cluster sandwich covariance, balanced pair-condition weights; AME delta-method intervals'}


def h3_summary(outcomes, iterations=2000):
    configs = sorted({r['config_id'] for r in outcomes})
    report = {}
    for sensitivity in [False, True]:
        subset = [r for r in outcomes if not sensitivity or r['without_gpt4']]
        tag = 'without_gpt4_candidates' if sensitivity else 'all'
        report[tag] = {}
        for n in [1, 2, 4]:
            for cfg in ['pooled'] + configs:
                rows = [r for r in subset if r['n'] == n and (cfg == 'pooled' or r['config_id'] == cfg)]
                # Pooled primary: retain a prompt only when all configurations have valid outcomes.
                # This prevents changing configuration weights when failures differ by model.
                if cfg == 'pooled':
                    from collections import Counter
                    complete = Counter(r['prompt_id'] for r in rows if not r['missing'])
                    good = {p for p, count in complete.items() if count == len(configs)}
                    selected = [r for r in rows if r['prompt_id'] in good]
                else:
                    selected = [r for r in rows if not r['missing']]
                report[tag][f'n={n}:{cfg}'] = {'planned_outcomes': len(rows), 'included_outcomes': len(selected),
                    'missing_outcomes': sum(r['missing'] for r in rows),
                    **{metric: clustered_interval(selected, metric, iterations) for metric in ['format_drift', 'regret', 'selected_quality']}}
    return report
