
def _read(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, dtype={A: str, O: str})


def _save(model, folder, results=None):
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    (p/'fit.json').write_text(json.dumps({'parameters': model.parameters, **model.diagnostics}, indent=2))
    model.states.to_csv(p/'organization_states.csv', index=False)
    model.account_scales.to_csv(p/'account_scales.csv', index=False)
    model.age_profile.rename('daily_revenue').to_csv(p/'other_age_profile.csv')
    model.new_profile.rename('daily_revenue').to_csv(p/'new_age_profile.csv')
    model.historical_arrivals.rename('arrivals').to_csv(p/'historical_arrivals.csv')
    model.price_template().to_csv(p/'required_external_prices.csv', index=False)
    if results:
        for name, table in results.items():
            table.to_csv(p/(name+'.csv'), index=False)


def _self_test():
    rows = []
    for i in range(16):
        landing = START + (i-12) if i >= 12 else START-12
        for m in TRAIN[TRAIN >= landing]:
            credits = 0. if i == 15 else float((20-i)*m.days_in_month)
            rows.append({A: f'a{i:02}', O: f'o{i:02}', T: str(m), C: credits, R: credits*2, 'cohort_age': m.ordinal-landing.ordinal})
    df = pd.DataFrame(rows)
    original = df.copy(deep=True)
    model = fit_model(df, complete_arrival_population=True)
    pd.testing.assert_frame_equal(df, original)
    np.testing.assert_allclose(model.states.level, 1.)
    np.testing.assert_allclose(model.states.trend, 0., atol=1e-12)
    prices = model.price_template().fillna({'effective_price': 2.})
    result = model.forecast(prices, no_separate_top_new_orgs=True)
    monthly = result['monthly']
    parts = monthly[monthly.component != 'total'].groupby(T).revenue_base.sum()
    totals = monthly[monthly.component == 'total'].set_index(T).revenue_base
    np.testing.assert_allclose(parts, totals)
    assert len(monthly) == 48 and np.isfinite(monthly.revenue_base).all()
    # Exact independent expectation: ten largest accounts, constant daily credits.
    org = result['organizations']
    expected = org[O].map({f'o{i:02}': 20-i for i in range(10)}) * org[T].dt.days_in_month
    np.testing.assert_allclose(org.credits_base, expected)
    adj = pd.DataFrame([{A: 'a00', T: str(FUTURE[0]), 'delta': .5}])
    adjusted = model.forecast(prices, adjustments=adj, no_separate_top_new_orgs=True)
    target = adjusted['organizations'].query('account__key == "a00"').iloc[0]
    assert np.isclose(target.revenue_adjusted, target.revenue_base*1.5)
    # Uniform age-zero rate maps back to observed per-arrival monthly dollars.
    counts = model.historical_arrivals
    assert counts.sum() == 4 and np.isclose(model.arrivals_per_month, 4/12)
    for broken in [pd.concat([df, df.iloc[[0]]]), df.drop(index=1), df.assign(cohort_age=-1)]:
        try:
            fit_model(broken, complete_arrival_population=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid training input accepted')
    try:
        model.forecast(prices.iloc[1:], no_separate_top_new_orgs=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing price accepted')
    # Explicit all-zero history: no numerical explosions or manufactured usage.
    zero = df.assign(**{C: 0., R: 0.})
    zm = fit_model(zero, complete_arrival_population=True)
    zr = zm.forecast(zm.price_template().fillna({'effective_price': 2.}), no_separate_top_new_orgs=True)
    assert (zr['monthly'].revenue_base == 0).all()
    print('Self-tests passed: level/trend, calendar exposure, partition totals, adjustments, zero histories, and input failures.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input')
    ap.add_argument('--out', default='fy28_results')
    ap.add_argument('--prices')
    ap.add_argument('--landing-register')
    ap.add_argument('--adjustments')
    ap.add_argument('--top-new-orgs')
    ap.add_argument('--arrival-scenario')
    ap.add_argument('--complete-arrival-population', action='store_true')
    ap.add_argument('--no-separate-top-new-orgs', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.input:
        ap.error('--input is required (or use --self-test)')
    model = fit_model(_read(args.input), complete_arrival_population=args.complete_arrival_population,
                      landing_register=_read(args.landing_register) if args.landing_register else None)
    results = None
    if args.prices:
        results = model.forecast(_read(args.prices),
            adjustments=_read(args.adjustments) if args.adjustments else None,
            top_new_orgs=_read(args.top_new_orgs) if args.top_new_orgs else None,
            no_separate_top_new_orgs=args.no_separate_top_new_orgs,
            arrival_scenario=_read(args.arrival_scenario) if args.arrival_scenario else None)
    _save(model, args.out, results)
    print(json.dumps(model.parameters, indent=2))
    print('Saved fit and ' + ('FY28 forecasts' if results else 'external price template; no total revenue forecast without prices') + ' to ' + args.out)


if __name__ == '__main__':
    main()
