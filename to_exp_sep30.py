"""Monthly forecast search, backtesting, horizon selection and Matplotlib plots.

Only NumPy/pandas are required to load results and compute MAE. Estimator and
plotting dependencies are imported on demand. Chronos runs in isolated processes.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile
from time import perf_counter
import warnings
import numpy as np
import pandas as pd

RESULT_COLUMNS = ['task_id', 'month', 'forecast_month', 'model_id', 'y', 'y_hat']
DEFAULT_TARGETS = ['dollar_usage','dollar_usage_adjusted','chc_usage','chc_usage_adjusted']
FEATURES = ['lag_1','lag_2','lag_3','lag_6','lag_12','mean_3','mean_6','std_3','month_sin','month_cos']
ALIASES = {'RFD':'random_forest','LASSOD':'lasso','change_xgboost':'xgb_changes','level_xgboost':'xgb_levels'}
LABELS = {'lasso':'Lasso','random_forest':'Random forest','xgb_changes':'XGBoost changes','xgb_levels':'XGBoost levels',
          'chronos_levels':'Chronos-2 levels','chronos_changes':'Chronos-2 changes',
          'chronos_finetuned':'Chronos-2 fine-tuned changes','drift':'3-month drift','trend':'6-month trend','last_value':'Last value'}


# Shared by forecast and error plots; independent of ranking and display order.
FAMILY_COLORS = {
    'lasso': 'blue',
    'random_forest': 'darkviolet',
    'xgb_changes': 'red',
    'xgb_levels': 'sienna',
    'chronos_levels': 'green',
    'chronos_changes': 'teal',
    'chronos_finetuned': 'darkorange',
    'drift': 'slategray',
    'trend': 'olive',
    'last_value': 'silver',
}
GLOBAL_FORECAST_COLOR = 'red'


def _family(model_id):
    for prefix, family in [('XGBD_','xgb_changes'),('XGB_','xgb_levels'),('RFD_','random_forest'),('LASSOD_','lasso'),
        ('CHRONOS2_DELTA_LORA','chronos_finetuned'),('CHRONOS2_DELTA_ZERO','chronos_changes'),
        ('CHRONOS2_C','chronos_levels'),('BASE_DRIFT','drift'),('BASE_TREND','trend'),('BASE_LAST','last_value')]:
        if model_id.startswith(prefix): return family
    raise ValueError(f'Unknown model family for {model_id}; supply specifications with a family.')


def _adjust(series):
    flags=(series-series.mean()).abs().gt(2*series.std(ddof=1))
    return series.mask(flags).interpolate(method='linear',limit_area='inside').fillna(series)


def _features(history, month):
    v=np.asarray(history,dtype=float)
    return np.array([*[v[-i] for i in (1,2,3,6,12)],v[-3:].mean(),v[-6:].mean(),v[-3:].std(ddof=1),
                     np.sin(2*np.pi*month.month/12),np.cos(2*np.pi*month.month/12)])


def _reconstruct_changes(changes,initial,nonnegative):
    """Reconstruct levels, applying the floor before the next change is added."""
    previous=float(initial);levels=[]
    for change in changes:
        previous=previous+float(change)
        if nonnegative:previous=max(0.,previous)
        levels.append(previous)
    return np.asarray(levels)


def _fit_candidate(spec, folds):
    """One model specification; each fold fits independently."""
    rows=[]; metadata=[]; family=spec['family']
    for fold in folds:
        history=np.asarray(fold['history'],dtype=float)
        dates=pd.to_datetime(fold['history_months']); months=pd.to_datetime(fold['months'])
        sx=sy=None; effective=len(history)
        if family in {'xgb_levels','xgb_changes','random_forest','lasso'}:
            x=np.vstack([_features(history[:i],dates[i]) for i in range(12,len(history))])
            target=history[12:] if family=='xgb_levels' else np.diff(history)[11:]
            effective=min(spec.get('window') or len(target),len(target))
            if effective<2: raise ValueError('At least two eligible supervised examples are required.')
            x=x[-effective:]; target=target[-effective:]
            if family.startswith('xgb'):
                from xgboost import XGBRegressor
                model=XGBRegressor(**spec['params']).fit(x,target)
            elif family=='random_forest':
                from sklearn.ensemble import RandomForestRegressor
                model=RandomForestRegressor(**spec['params']).fit(x,target)
            else:
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LassoLars
                from sklearn.exceptions import ConvergenceWarning
                sx=StandardScaler().fit(x); sy=StandardScaler().fit(target[:,None])
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    model=LassoLars(**spec['params']).fit(sx.transform(x),sy.transform(target[:,None]).ravel())
        previous=list(history)
        for h,month in enumerate(months,1):
            if family=='last_value': pred=history[-1]
            elif family=='drift': pred=history[-1]+h*(history[-1]-history[-4])/3
            elif family=='trend':
                slope,intercept=np.polyfit(np.arange(6),history[-6:],1);pred=intercept+slope*(5+h)
            else:
                x=_features(previous,month)[None,:]
                value=float(sy.inverse_transform(model.predict(sx.transform(x))[:,None])[0,0]) if sx is not None else float(model.predict(x)[0])
                pred=value if family=='xgb_levels' else previous[-1]+value
            if not np.isfinite(pred): raise ValueError(f'Non-finite forecast: {spec["model_id"]}')
            unbounded=float(pred)
            if fold.get('nonnegative',False):pred=max(0.,unbounded)
            rows.append(dict(task_id=fold['task_id'],month=month,model_id=spec['model_id'],y=fold['actuals'][h-1],y_hat=float(pred)))
            metadata.append(dict(model_id=spec['model_id'],month=month,family=family,backtest_id=fold['backtest_id'],
                forecast_origin=fold['origin'],horizon=h,nonnegative=fold.get('nonnegative',False),unbounded_y_hat=unbounded,was_clipped=float(pred)!=unbounded,requested_window=spec.get('window'),effective_training_rows=effective))
            previous.append(float(pred))
    return rows,metadata


def _timed_fit_candidate(spec, folds):
    """Measure actual worker execution, excluding time waiting in the queue."""
    started=perf_counter()
    outcome=_fit_candidate(spec,folds)
    return outcome,started,perf_counter()


def _chronos_worker(request_path, response_path):
    """Isolated from sklearn/XGBoost native runtimes; variable horizon throughout."""
    import torch
    from chronos import Chronos2Pipeline
    from transformers import TrainerCallback
    request=json.loads(Path(request_path).read_text());spec=request['spec'];family=spec['family']
    horizon=request['horizon']; model_path=request['model_path']
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    base=Chronos2Pipeline.from_pretrained(model_path,device_map='cpu',dtype=torch.float32,local_files_only=True)
    rows=[];metadata=[]
    class Log(TrainerCallback):
        def __init__(self):self.records=[]
        def on_log(self,args,state,control,logs=None,**kwargs):self.records.append({'step':state.global_step,**(logs or {})})
    for fold in request['folds']:
        values=np.asarray(fold['history'],dtype=float); dates=pd.to_datetime(fold['history_months'])
        is_delta=family!='chronos_levels'
        if is_delta: values=np.diff(values);dates=dates[1:]
        count=min(spec.get('context') or len(values),len(values))
        pipeline=base
        if family=='chronos_finetuned':
            params=spec.get('params',{});min_past=int(params.get('min_past',6))
            if len(values)<horizon+min_past:
                raise ValueError('Chronos fine-tuning needs at least horizon + min_past historical changes.')
            torch.manual_seed(42);np.random.seed(42);log=Log()
            folder=Path(request['checkpoint_dir'])/spec['model_id']/fold['backtest_id']
            pipeline=base.fit(inputs=[values.astype(np.float32)],prediction_length=horizon,finetune_mode='lora',
                context_length=spec.get('context') or len(values),min_past=min_past,
                learning_rate=params.get('learning_rate',1e-5),num_steps=params.get('num_steps',100),
                batch_size=params.get('batch_size',4),output_dir=folder,callbacks=[log],
                remove_printer_callback=True,disable_tqdm=True,logging_steps=25,optim='adamw_torch',seed=42,data_seed=42)
            (folder/'training_log.json').write_text(json.dumps(log.records))
        frame=pd.DataFrame({'series':'series','month':dates[-count:],'target':values[-count:]})
        pred=pipeline.predict_df(frame,prediction_length=horizon,quantile_levels=[.1,.5,.9],id_column='series',
            timestamp_column='month',target='target',freq='MS',context_length=count,cross_learning=False,batch_size=1).sort_values('month')
        assert pd.DatetimeIndex(pred.month).equals(pd.DatetimeIndex(fold['months']))
        levels=pred['0.5'].to_numpy()
        if is_delta: levels=_reconstruct_changes(levels,fold['history'][-1],fold.get('nonnegative',False))
        for h,(month,level) in enumerate(zip(pred.month,levels),1):
            if fold.get('nonnegative',False):level=max(0.,float(level))
            rows.append(dict(task_id=fold['task_id'],month=month,model_id=spec['model_id'],y=fold['actuals'][h-1],y_hat=float(level)))
            metadata.append(dict(model_id=spec['model_id'],month=month,family=family,backtest_id=fold['backtest_id'],
                forecast_origin=fold['origin'],horizon=h,effective_context_months=count,nonnegative=fold.get('nonnegative',False)))
        if pipeline is not base:
            del pipeline
            import gc
            gc.collect()
    Path(response_path).write_text(json.dumps([rows,metadata],default=str,allow_nan=False))


class _FinancialReports:
    """Monthly and February-to-January fiscal reports for a forecaster."""

    def __init__(self,model):
        self._model=model

    def __call__(self,target=None):
        """Keep the original fin_report() syntax as a monthly-report alias."""
        return self.monthly(target=target)

    def monthly(self,target=None):
        """Observed and selected forecast revenue with calendar normalization."""
        return self._model._fin_report_monthly(target=target)

    def fy(self,target=None):
        """Sum months by fiscal ending year: FY28 is Feb 2027–Jan 2028.

        source is actual, forecast, or mixed. Incomplete fiscal years contain
        available months only; months and is_complete expose that coverage.
        rev_adjusted sums the monthly 30-day-normalized revenue values.
        revenue_change and pct_change compare total revenue with the preceding full FY.
        pct_change is in percent (25 means 25%); zero denominators yield NaN.
        """
        monthly=self.monthly(target=target).copy()
        monthly['fy_year']=monthly.month.dt.year+monthly.month.dt.month.ge(2).astype(int)
        rows=[]
        for year,group in monthly.groupby('fy_year',sort=True):
            actual=group.source.eq('actual');forecast=group.source.eq('forecast')
            months=group.month.nunique()
            source='mixed' if actual.any() and forecast.any() else ('actual' if actual.any() else 'forecast')
            rows.append(dict(target=group.target.iloc[0],fy=f'FY{year%100:02d}',
                fy_start=pd.Timestamp(year=int(year)-1,month=2,day=1),
                fy_end=pd.Timestamp(year=int(year),month=1,day=31),source=source,
                revenue=group.revenue.sum(),rev_adjusted=group.rev_adjusted.sum(),
                actual_revenue=group.loc[actual,'revenue'].sum(),
                forecast_revenue=group.loc[forecast,'revenue'].sum(),
                months=months,actual_months=int(actual.sum()),forecast_months=int(forecast.sum()),
                is_complete=months==12))
        report=pd.DataFrame(rows)
        previous=report.revenue.shift(1)
        comparable=(report.is_complete & report.is_complete.shift(1,fill_value=False)
                    & report.fy_end.dt.year.diff().eq(1))
        revenue_change=(report.revenue-previous).where(comparable)
        percent=(revenue_change/previous.where(previous.ne(0))*100)
        position=report.columns.get_loc('revenue')+1
        report.insert(position,'revenue_change',revenue_change)
        report.insert(position+1,'pct_change',percent)
        return report


class MonthlyForecaster:
    """Forecast monthly levels, select configurations per horizon, and retain backtests.

    Args:
        data: daily CSV path, daily DataFrame, or monthly Series with DatetimeIndex.
        target: None (default) forecasts all four targets independently. A target
            string restricts to one; a list selects a subset. Adjusted targets
            train on adjusted histories but always score against raw observations.
        horizon: positive number of months to forecast.
        backtest_origins: training counts or last-training-month dates. Optional.
        n_backtests: number of automatic chronological backtests (default 5).
            Explicit backtest_origins override this count.
        specs: candidate specification dictionaries; defaults to the saved full grid.
        families: optional subset of family names.
        loss: MAE, MSE, RMSE, MSPE, or MAPE (case-insensitive); selects winners.
            This is the backtest selection metric, not the estimator training objective.
        nonnegative: floor predicted levels at zero in every fold and final forecast.
            Recursive features receive the constrained prediction. Default True.
    """
    def __init__(self,data,target=None,horizon=6,backtest_origins=None,n_backtests=5,
                 specs=None,families=None,workers=4,output_dir='forecast_outputs',
                 catalog_path=None,chronos_model_path=None,chronos_python=None,loss="MAE",nonnegative=True):
        if isinstance(horizon,bool) or not isinstance(horizon,int) or horizon<1:
            raise ValueError('horizon must be a positive integer.')
        if not isinstance(nonnegative,bool):raise ValueError("nonnegative must be True or False.")
        self.nonnegative=nonnegative
        self.loss=self._validate_loss(loss)
        self.horizon=horizon;self.target=target;self.workers=max(1,int(workers));self.output_dir=Path(output_dir)
        self.chronos_python=Path(chronos_python) if chronos_python else None
        self._chronos_python_resolved=None
        self.models_ = None
        if target is None or isinstance(target,(list,tuple)):
            targets=list(DEFAULT_TARGETS if target is None else target)
            if not targets or len(set(targets))!=len(targets) or not set(targets).issubset(DEFAULT_TARGETS):
                raise ValueError('Supply unique targets from dollar_usage, chc_usage and their adjusted variants.')
            if isinstance(data,pd.Series):raise ValueError('Specify a single target when passing a monthly Series.')
            frame=pd.read_csv(data) if isinstance(data,(str,Path)) else data.copy()
            self.targets_=targets
            self.models_={name:MonthlyForecaster(frame,target=name,horizon=horizon,backtest_origins=backtest_origins,
                n_backtests=n_backtests,specs=specs,families=families,workers=workers,output_dir=self.output_dir/name,
                catalog_path=catalog_path,chronos_model_path=chronos_model_path,chronos_python=chronos_python,loss=self.loss,nonnegative=self.nonnegative) for name in targets}
            first=next(iter(self.models_.values()))
            self.origins=first.origins;self.n_backtests=first.n_backtests
            self.selection_=None;self.selection_scope_=None
            self._sync_targets()
            return
        self.targets_=[target]
        self.adjusted=target.endswith('_adjusted');self.raw_target=target.removesuffix('_adjusted')
        self.raw=self._load_data(data)
        self.actuals=self.raw.copy()  # Evaluation always uses original observed usage.
        self.adjusted_history=_adjust(self.raw)  # Diagnostic only; never used to build backtest training folds.
        self.origins=self._origins(backtest_origins,n_backtests)
        self.n_backtests=len(self.origins)
        self.chronos_model_path=Path(chronos_model_path or Path(__file__).resolve().parent.parent/'work'/'chronos-2-model')
        self.specs=self._specs(specs,catalog_path)
        if families is not None:self.specs=[s for s in self.specs if s['family'] in families]
        if not self.specs:raise ValueError('No candidate specifications remain.')
        ids=[s['model_id'] for s in self.specs]
        if len(ids)!=len(set(ids)):raise ValueError('model_id must be unique.')
        for spec in self.specs:
            if spec['family']=='chronos_finetuned':
                required=self.horizon+int(spec.get('params',{}).get('min_past',6))+1
                if min(self.origins)<required:raise ValueError(f'Chronos fine-tuning needs at least {required} training months for this horizon.')
        self.results_=pd.DataFrame(columns=RESULT_COLUMNS);self.metadata_=pd.DataFrame()
        self.selection_=None;self.selection_scope_=None

    def _annotate_metadata(self):
        """Keep target and actual values aligned with prediction rows."""
        meta=self.metadata_.drop(columns=['target','y_actual'],errors='ignore').copy()
        targets=self.results_['target'].to_numpy() if self.models_ is not None else self.target
        meta.insert(0,'target',targets)
        meta['y_actual']=self.results_['y'].to_numpy()
        if self.models_ is None and 'month' in self.results_:
            meta['y_adjusted']=self.adjusted_history.reindex(self.results_.month).to_numpy()
        self.metadata_=meta

    def _sync_targets(self):
        rows=[];metadata=[];offset=0
        for target,model in self.models_.items():
            rows.append(model.results_.assign(target=target))
            if not model.metadata_.empty:
                meta=model.metadata_.copy();meta['target']=target
                meta['row_id']=np.arange(offset,offset+len(meta));metadata.append(meta)
            offset+=len(model.results_)
        self.results_=pd.concat(rows,ignore_index=True)
        self.metadata_=pd.concat(metadata,ignore_index=True) if metadata else pd.DataFrame()
        self._annotate_metadata()

    def _target_tables(self,method,*args,**kwargs):
        return pd.concat([getattr(model,method)(*args,**kwargs).assign(target=target)
                          for target,model in self.models_.items()],ignore_index=True)

    def _load_data(self,data):
        if isinstance(data,pd.Series):
            raw=data.copy().sort_index()
            if not isinstance(raw.index,pd.DatetimeIndex):raise ValueError('Monthly Series needs a DatetimeIndex.')
            raw.index=raw.index.to_period('M').to_timestamp()
        else:
            df=pd.read_csv(data) if isinstance(data,(str,Path)) else data.copy()
            if 'service__product' in df:df=df[df['service__product'].eq('clickhouse')]
            df['usage_day']=pd.to_datetime(df.usage_day)
            groups=df.groupby(df.usage_day.dt.to_period('M'));coverage=groups.usage_day.nunique()
            raw=groups[self.raw_target].sum().loc[coverage.eq(coverage.index.days_in_month)]
            raw.index=raw.index.to_timestamp()
        if len(raw)<14 or raw.index.has_duplicates or not np.isfinite(raw.to_numpy(dtype=float)).all():
            raise ValueError('Need at least 14 finite, unique monthly observations.')
        if not raw.index.equals(pd.date_range(raw.index.min(),raw.index.max(),freq='MS')):
            raise ValueError('Monthly history must be contiguous.')
        raw.index.name='month';return raw.astype(float)

    def _origins(self,origins,n_backtests):
        if origins is None:
            if isinstance(n_backtests,bool) or not isinstance(n_backtests,int) or n_backtests<5:
                raise ValueError('At least five backtests are required: four or more for selection and the latest for evaluation.')
            latest=len(self.raw)-self.horizon
            earliest=max(14,self.horizon+7)
            if latest<earliest+n_backtests-1:raise ValueError('Insufficient history for the requested horizon/backtests.')
            origins=np.linspace(earliest,latest,n_backtests,dtype=int).tolist()
        result=[]
        for origin in origins:
            count=int(origin) if isinstance(origin,(int,np.integer)) else int(self.raw.index.searchsorted(pd.Timestamp(origin).to_period('M').to_timestamp(),side='right'))
            if count<14 or count+self.horizon>len(self.raw):raise ValueError('Each origin needs 14 training months and a complete forecast horizon.')
            result.append(count)
        if len(result)<5 or result!=sorted(set(result)):raise ValueError('Supply at least five distinct chronological origins.')
        return result

    def _specs(self,specs,catalog_path):
        if specs is None:
            path=Path(catalog_path or Path(__file__).with_name('forecast_model_catalog.json'))
            specs=json.loads(path.read_text())
            specs += [{'model_id':f'CHRONOS2_C{c or "ALL"}','family':'chronos_levels','context':c,'params':{}} for c in [12,24,36,None]]
            specs += [{'model_id':'CHRONOS2_DELTA_ZERO_C36','family':'chronos_changes','context':36,'params':{}},
                      {'model_id':'CHRONOS2_DELTA_LORA_C36_S100','family':'chronos_finetuned','context':36,'params':{'num_steps':100,'learning_rate':1e-5,'batch_size':4,'min_past':6}}]
            specs += [{'model_id':mid,'family':f,'params':{}} for mid,f in [('BASE_LAST_VALUE','last_value'),('BASE_DRIFT_3','drift'),('BASE_TREND_6','trend')]]
        normalized=[]
        for source in specs:
            s=json.loads(json.dumps(source));s['family']=ALIASES.get(s['family'],s['family'])
            if s['family']=='xgb_changes':s['model_id']=s['model_id'].replace('XGB_','XGBD_',1)
            if s['family'] not in LABELS:raise ValueError(f'Unknown family: {s["family"]}')
            if s.get('window') is not None and (not isinstance(s['window'],int) or s['window']<2):raise ValueError('Training window must be >=2 or None.')
            normalized.append(s)
        return normalized

    def _fold(self,count,name):
        history=self.raw.iloc[:count];history=_adjust(history) if self.adjusted else history
        months=pd.date_range(history.index[-1]+pd.offsets.MonthBegin(),periods=self.horizon,freq='MS')
        final=name=='final'
        return dict(backtest_id=name,task_id='final_estimate' if final else 'back_test',origin=str(history.index[-1]),
                    history=history.tolist(),history_months=[str(m) for m in history.index],months=[str(m) for m in months],
                    actuals=[None]*self.horizon if final else self.actuals.reindex(months).tolist(),nonnegative=self.nonnegative)

    def _resolve_chronos_python(self):
        """Validate an isolated Chronos runtime before expensive classical fits."""
        if self._chronos_python_resolved is not None:return self._chronos_python_resolved
        local=Path(__file__).resolve().parent.parent/'work'/'xgb-env'/'bin'/'python'
        candidates=[self.chronos_python] if self.chronos_python else [local,Path(sys.executable)]
        needs_lora=any(s['family']=='chronos_finetuned' for s in self.specs)
        probe='from chronos import Chronos2Pipeline; import torch, pandas, numpy'
        if needs_lora:probe+='; import peft, accelerate'
        failures=[]
        for python in dict.fromkeys(candidates):
            if not python.is_file():
                failures.append(f'{python}: interpreter not found');continue
            try:
                result=subprocess.run([str(python),'-c',probe],capture_output=True,text=True,timeout=90)
            except (OSError,subprocess.TimeoutExpired) as exc:
                failures.append(f'{python}: {exc}');continue
            if result.returncode==0:
                if not self.chronos_model_path.is_dir():raise RuntimeError(f'Chronos model directory not found: {self.chronos_model_path}')
                self._chronos_python_resolved=str(python)
                return str(python)
            failures.append(f'{python}: {result.stderr.strip()[-1200:]}')
        raise RuntimeError('No working Chronos Python environment. Pass chronos_python="/path/to/python" '
            'with chronos-forecasting, torch, transformers, accelerate and peft installed.\n'+'\n'.join(failures))

    def _run(self,folds):
        chronos_python=self._resolve_chronos_python() if any(s['family'].startswith('chronos') for s in self.specs) else None
        outcomes={};classical=[s for s in self.specs if not s['family'].startswith('chronos')]
        phase=f'backtests ({len(folds)})' if folds[0]['task_id']=='back_test' else 'final forecast'
        remaining={family:sum(s['family']==family for s in self.specs) for family in dict.fromkeys(s['family'] for s in self.specs)}
        counts=remaining.copy();timings={}
        def record_timing(spec,started,finished):
            family=spec['family']
            first,last=timings.get(family,(started,finished))
            timings[family]=(min(first,started),max(last,finished))
            remaining[family]-=1
            if remaining[family]==0:
                first,last=timings[family]
                print(f'[{self.target} | {phase}] {LABELS[family]}: {last-first:.2f} s elapsed '
                      f'({counts[family]} specifications)',flush=True)
        if self.workers==1:
            for s in classical:
                outcome,started,finished=_timed_fit_candidate(s,folds)
                outcomes[s['model_id']]=outcome;record_timing(s,started,finished)
        else:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                futures={pool.submit(_timed_fit_candidate,s,folds):s for s in classical}
                for future in as_completed(futures):
                    spec=futures[future];outcome,started,finished=future.result()
                    outcomes[spec['model_id']]=outcome;record_timing(spec,started,finished)
        for s in self.specs:
            if not s['family'].startswith('chronos'):continue
            started=perf_counter()
            self.output_dir.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory() as folder:
                request=Path(folder)/'request.json';response=Path(folder)/'response.json'
                request.write_text(json.dumps(dict(spec=s,folds=folds,horizon=self.horizon,model_path=str(self.chronos_model_path),checkpoint_dir=str(self.output_dir/'checkpoints'))))
                subprocess.run([chronos_python,str(Path(__file__).resolve()),'--chronos-worker',str(request),str(response)],check=True)
                outcomes[s['model_id']]=json.loads(response.read_text())
            record_timing(s,started,perf_counter())
        rows=[];meta=[]
        for s in self.specs:
            a,b=outcomes[s['model_id']];rows.extend(a);meta.extend(b)
        result=pd.DataFrame(rows,columns=RESULT_COLUMNS);metadata=pd.DataFrame(meta)
        result['forecast_month']=metadata['horizon'].astype(int).to_numpy()
        result.month=pd.to_datetime(result.month);result.y=pd.to_numeric(result.y)
        metadata.month=pd.to_datetime(metadata.month);metadata.forecast_origin=pd.to_datetime(metadata.forecast_origin)
        if not np.isfinite(result.y_hat).all():raise ValueError('Non-finite predictions.')
        return result,metadata

    def _replace(self,task_id,result,meta):
        if not self.results_.empty:
            keep=self.results_.task_id.ne(task_id)
            result=pd.concat([self.results_[keep],result],ignore_index=True)
            meta=pd.concat([self.metadata_.loc[keep].drop(columns='row_id'),meta],ignore_index=True)
        self.results_=result.reset_index(drop=True);self.metadata_=meta.reset_index(drop=True)
        self.metadata_.insert(0,'row_id',np.arange(len(meta)))
        self._annotate_metadata()
        self.selection_=None;self.selection_scope_=None

    def backtest(self):
        """Fit every specification at every configured origin; return long predictions."""
        if self.models_ is not None:
            result=self._target_tables('backtest');self._sync_targets()
            self.selection_=None;self.selection_scope_=None
            return result
        folds=[self._fold(n,f'bt_{i:02d}') for i,n in enumerate(self.origins,1)]
        result,meta=self._run(folds);self._replace('back_test',result,meta)
        return result

    def back(self):
        """Alias for backtest()."""
        return self.backtest()

    def train(self,run_backtests=True):
        """Fit final forecasts; optionally recompute all backtests first. Return self."""
        if self.models_ is not None:
            for target,model in self.models_.items():
                print(f'Training {target} ({self.horizon}-month forecast, {self.n_backtests} backtests)',flush=True)
                model.train(run_backtests=run_backtests)
            self._sync_targets();self.selection_=None;self.selection_scope_=None
            return self
        if run_backtests:self.backtest()
        result,meta=self._run([self._fold(len(self.raw),'final')]);self._replace('final_estimate',result,meta)
        return self

    def _joined(self):
        if self.results_.empty:raise ValueError('Run train/backtest or load_results first.')
        return self.results_.join(self.metadata_.drop(columns=['row_id',*self.results_.columns],errors='ignore'))

    @staticmethod
    def _validate_loss(loss):
        name=str(loss).strip().upper()
        if name not in {'MAE','MSE','RMSE','MSPE','MAPE'}:
            raise ValueError('loss must be MAE, MSE, RMSE, MSPE, or MAPE.')
        return name

    def metrics(self,folds=None,loss=None):
        """Score each model/family/horizon using the configured or specified loss.

        MAPE is mean absolute percentage error (%). MSPE is mean squared
        percentage error (% squared). Both raise for zero actuals, rather than
        silently dropping observations or adding an arbitrary denominator.
        """
        name=self._validate_loss(self.loss if loss is None else loss)
        if self.models_ is not None:return self._target_tables('metrics',folds=folds,loss=name)
        data=self._joined();bt=data[data.task_id.eq('back_test')].copy()
        available=sorted(bt.backtest_id.unique())
        if not available:raise ValueError('No backtests to score.')
        if isinstance(folds,str):
            if folds=='selection':folds=available[:-1]
            elif folds=='evaluation':folds=available[-1:]
            else:raise ValueError('folds must be selection, evaluation, or a list.')
        if folds is not None:
            if not set(folds).issubset(available) or not len(folds):raise ValueError('Unknown or empty folds.')
            bt=bt[bt.backtest_id.isin(folds)]
        error=bt.y_hat-bt.y
        if name in {'MAPE','MSPE'}:
            if bt.y.eq(0).any():
                raise ValueError(f'{name} is undefined for zero actuals. Choose MAE, MSE, or RMSE.')
            error=100*error/bt.y.abs()
        bt['error_value']=error.abs() if name in {'MAE','MAPE'} else error.pow(2)
        if not np.isfinite(bt.error_value).all():raise ValueError('Non-finite errors for the requested loss.')
        scores=bt.groupby(['model_id','family','horizon']).agg(value=('error_value','mean'),n_backtests=('error_value','size')).reset_index()
        if name=='RMSE':scores['value']=np.sqrt(scores.value)
        return scores.rename(columns={'horizon':'forecast_month','value':name.lower()})

    def mae(self,folds=None):
        """Always report MAE, independently of the model-selection loss."""
        return self.metrics(folds=folds,loss='MAE')

    def MAE(self,folds=None):
        """Alias for mae()."""
        return self.mae(folds)

    def select(self,scope='family',folds='selection'):
        """Lowest configured loss per family/horizon, or across all families."""
        if self.models_ is not None:
            self.selection_=self._target_tables('select',scope=scope,folds=folds);self.selection_scope_=scope
            return self.selection_.copy()
        if scope not in ['family','global']:raise ValueError('scope must be family or global.')
        scores=self.metrics(folds);keys=['family','forecast_month'] if scope=='family' else ['forecast_month']
        self.selection_=scores.sort_values([*keys,self.loss.lower(),'model_id']).groupby(keys,as_index=False).first()
        if scope=='global':self.selection_.insert(0,'target',self.target)
        self.selection_scope_=scope
        return self.selection_.copy()

    def forecast(self,scope='global'):
        """Assemble selected horizon predictions without cross-model recursive inputs."""
        if self.models_ is not None:return self._target_tables('forecast',scope=scope)
        if self.selection_ is None or self.selection_scope_!=scope:self.select(scope)
        final=self._joined();final=final[final.task_id.eq('final_estimate')].drop(columns='horizon')
        if final.empty:raise ValueError('Run train() for final forecasts.')
        return self.selection_.merge(final[['model_id','family','forecast_month','month','y_hat']],on=['model_id','family','forecast_month'],validate='one_to_one').sort_values(['family','forecast_month'] if scope=='family' else ['forecast_month'])

    @property
    def fin_report(self):
        """Access fin_report.monthly() or fin_report.fy()."""
        return _FinancialReports(self)

    def _fin_report_monthly(self,target=None):
        """Return observed and forecast revenue, 30-day revenue and MoM growth.

        Uses the horizon-specific global forecast. Multi-target models default
        to dollar_usage (or dollar_usage_adjusted if that is the only revenue
        target); target can explicitly choose either trained revenue variant.
        History is always original observed revenue. rev_adjusted is calendar
        normalization, not outlier adjustment. mom is a fraction (0.05 = 5%).
        The first row and rows following zero revenue have undefined MoM (NaN).
        """
        revenue_targets=('dollar_usage','dollar_usage_adjusted')
        if self.models_ is not None:
            chosen=target if target is not None else next((t for t in revenue_targets if t in self.models_),None)
            if chosen not in revenue_targets or chosen not in self.models_:
                raise ValueError('fin_report requires a trained dollar_usage or dollar_usage_adjusted target.')
            return self.models_[chosen]._fin_report_monthly()
        if self.raw_target!='dollar_usage' or (target is not None and target!=self.target):
            raise ValueError('fin_report requires this model to target dollar_usage or dollar_usage_adjusted.')
        forecasts=self.forecast(scope='global').sort_values('forecast_month')
        observed=pd.DataFrame({'month':self.actuals.index,'source':'actual','revenue':self.actuals.to_numpy()})
        future=forecasts[['month','y_hat']].rename(columns={'y_hat':'revenue'}).assign(source='forecast')
        report=pd.concat([observed,future],ignore_index=True).sort_values('month').reset_index(drop=True)
        report.insert(0,'target',self.target)
        report['days_in_month']=report.month.dt.days_in_month
        report['rev_adjusted']=report.revenue/report.days_in_month*30
        previous=report.rev_adjusted.shift(1)
        report['mom']=report.rev_adjusted/previous.where(previous.ne(0))-1
        return report[['target','month','source','revenue','days_in_month','rev_adjusted','mom']]

    def load_results(self,results,metadata):
        """Import saved predictions and score against original observed usage.

        Legacy adjusted labels are accepted only if they match this series' old
        full-history adjustment, then replaced with raw labels. Predictions stay
        unchanged; metrics and horizon selections are recomputed from raw labels.
        """
        if self.models_ is not None:
            if not isinstance(results,dict) or not isinstance(metadata,dict):
                raise ValueError('For multiple targets, pass dictionaries mapping each target to its results/metadata file.')
            if set(results)!=set(self.models_) or set(metadata)!=set(self.models_):raise ValueError('Result paths must cover every target.')
            for target,model in self.models_.items():model.load_results(results[target],metadata[target])
            self._sync_targets();self.selection_=None;self.selection_scope_=None
            return self
        result=pd.read_parquet(results) if str(results).endswith('.parquet') else pd.read_csv(results)
        meta=pd.read_csv(metadata,low_memory=False).set_index('row_id').reindex(result.index)
        if meta.model_id.isna().any() or not meta.model_id.equals(result.model_id):raise ValueError('Metadata/model alignment mismatch.')
        result.month=pd.to_datetime(result.month).astype('datetime64[ns]')
        meta.month=pd.to_datetime(meta.month,format='mixed').astype('datetime64[ns]')
        if not result.month.equals(meta.month):raise ValueError('Metadata/month alignment mismatch.')
        spec_family={s['model_id']:s['family'] for s in self.specs}
        if not set(result.model_id).issubset(spec_family):raise ValueError('Results contain models absent from specifications.')
        meta['family']=meta.model_id.map(spec_family)
        if self.nonnegative and ('nonnegative' not in meta or not meta.nonnegative.eq(True).all()):
            raise ValueError('Saved results were not generated with nonnegative=True. Retrain; do not clip old results after selection.')
        if self.nonnegative and result.y_hat.lt(0).any():raise ValueError('Saved nonnegative forecasts contain negatives.')
        columns=['model_id','month','family','backtest_id','forecast_origin','horizon']
        if 'nonnegative' in meta:columns.append('nonnegative')
        meta=meta[columns].copy()
        meta.forecast_origin=pd.to_datetime(meta.forecast_origin,format='mixed').astype('datetime64[ns]')
        groups=meta.groupby(['model_id','backtest_id']).horizon.agg(list)
        if not groups.map(lambda h:sorted(h)==list(range(1,self.horizon+1))).all():raise ValueError('Saved horizon does not match configured horizon.')
        expected={f'bt_{i:02d}':self.raw.index[n-1] for i,n in enumerate(self.origins,1)}
        expected['final']=self.raw.index[-1]
        if not meta.forecast_origin.equals(meta.backtest_id.map(expected).astype('datetime64[ns]')):raise ValueError('Saved origins do not match configured backtests.')
        bt=result.task_id.eq('back_test')
        observed=self.actuals.reindex(result.loc[bt,'month']).to_numpy()
        if not np.allclose(result.loc[bt,'y'],observed):
            legacy=self.adjusted_history.reindex(result.loc[bt,'month']).to_numpy()
            if not self.adjusted or not np.allclose(result.loc[bt,'y'],legacy):
                raise ValueError('Saved actuals do not match configured target.')
            warnings.warn('Replacing legacy adjusted backtest labels with observed usage; recompute selections and save results.',UserWarning)
        result.loc[bt,'y']=observed
        if not np.isfinite(result.y_hat).all():raise ValueError('Invalid predictions.')
        if not result.loc[~bt,'y'].isna().all():raise ValueError('Future y must be missing.')
        if 'forecast_month' in result and not np.array_equal(result.forecast_month.to_numpy(),meta.horizon.to_numpy()):
            raise ValueError('Saved forecast_month does not match metadata horizon.')
        result['forecast_month']=meta.horizon.astype(int).to_numpy()
        self.results_=result[RESULT_COLUMNS].reset_index(drop=True);self.metadata_=meta.reset_index(drop=True)
        self.metadata_.insert(0,'row_id',np.arange(len(meta)))
        self._annotate_metadata()
        self.selection_=None;self.selection_scope_=None
        return self

    def save(self,directory=None):
        """Persist predictions, source metadata, all MAEs, specifications and settings."""
        folder=Path(directory or self.output_dir);folder.mkdir(parents=True,exist_ok=True)
        if self.models_ is not None:
            for target,model in self.models_.items():model.save(folder/target)
            self._sync_targets()
            self.results_.to_csv(folder/'results_all_targets.csv',index=False)
            self.metadata_.to_csv(folder/'metadata_all_targets.csv',index=False)
            self.mae().to_csv(folder/'mae_target_model_family_month.csv',index=False)
            self.metrics().to_csv(folder/'loss_target_model_family_month.csv',index=False)
            self.select('global').to_csv(folder/'global_selections_all_targets.csv',index=False)
            if self.results_.task_id.eq('final_estimate').any():self.forecast('global').to_csv(folder/'global_forecasts_all_targets.csv',index=False)
            (folder/'settings.json').write_text(json.dumps({'targets':self.targets_,'horizon':self.horizon,'n_backtests':self.n_backtests,'origins':self.origins,'loss':self.loss,'nonnegative':self.nonnegative,'evaluation_actuals':'original observed usage'},indent=2))
            return folder
        self.results_.to_csv(folder/'results_long.csv',index=False);self.metadata_.to_csv(folder/'prediction_metadata.csv',index=False)
        self.mae().to_csv(folder/'mae_model_id_family_month.csv',index=False)
        self.mae('selection').to_csv(folder/'selection_mae.csv',index=False)
        self.mae('evaluation').to_csv(folder/'evaluation_mae.csv',index=False)
        self.metrics().to_csv(folder/'loss_model_id_family_month.csv',index=False)
        self.metrics('selection').to_csv(folder/'selection_loss.csv',index=False)
        self.metrics('evaluation').to_csv(folder/'evaluation_loss.csv',index=False)
        for scope in ['family','global']:
            self.select(scope).to_csv(folder/f'{scope}_selections.csv',index=False)
            if self.results_.task_id.eq('final_estimate').any():self.forecast(scope).to_csv(folder/f'{scope}_forecast.csv',index=False)
        (folder/'specifications.json').write_text(json.dumps(self.specs,indent=2))
        (folder/'settings.json').write_text(json.dumps({'target':self.target,'horizon':self.horizon,'backtest_origins':self.origins,'n_backtests':self.n_backtests,
            'features':FEATURES,'loss':self.loss,'nonnegative':self.nonnegative,'selection':'all but last backtest; separate evaluation on last, previously inspected historical data',
            'evaluation_actuals':self.raw_target,
            'adjustment':'origin-only training adjustments; original observed evaluation labels' if self.adjusted else 'none'},indent=2))
        return folder

    def plot(self,kind='forecast',scope='family',folds=None,backtest_lines=0,path=None,target=None):
        """Return a Matplotlib Figure; optionally save. kind='forecast' or 'mae'.

        Forecast gray lines show top XGBoost-change candidates ranked on selection
        folds. Backtest lines are hidden by default. Pass None to show ten
        candidates across all backtests, or a positive multiple of the backtest
        count to choose a specific number of lines.
        """
        if self.models_ is not None:
            if target is not None and target not in self.models_:raise ValueError('Unknown plot target.')
            figures={}
            for name,model in self.models_.items():
                if target is not None and name!=target:continue
                output=Path(path).with_name(f'{Path(path).stem}_{name}{Path(path).suffix}') if path and target is None else path
                figures[name]=model.plot(kind=kind,scope=scope,folds=folds,backtest_lines=backtest_lines,path=output)
            return figures[target] if target is not None else figures
        if target is not None and target!=self.target:raise ValueError('Unknown plot target.')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter
        selection=self.select(scope)
        if kind in {'mae','loss','metrics'}:
            name='MAE' if kind=='mae' else self.loss
            metric=name.lower();scale=1 if name in {'MAPE','MSPE'} else 1000
            unit='%' if name=='MAPE' else ('% squared' if name=='MSPE' else 'thousands')
            scores=selection[['model_id','family','forecast_month']].merge(self.metrics(folds,loss=name),on=['model_id','family','forecast_month'])
            scores['label']=scores.family.map(LABELS) if scope=='family' else 'Best across families'
            table=scores.pivot(index='label',columns='forecast_month',values=metric)
            table=table.loc[table.mean(axis=1).sort_values().index]
            fig,axes=plt.subplots(1,2,figsize=(16,6),layout='constrained')
            label_colors={LABELS[family]:color for family,color in FAMILY_COLORS.items()}
            for label,row in table.iterrows():
                color=GLOBAL_FORECAST_COLOR if scope=='global' else label_colors[label]
                axes[0].plot(table.columns,row,marker='o',label=label,color=color)
            axes[0].legend(fontsize=8);axes[0].set_xlabel('Forecast month');axes[0].set_ylabel(name);axes[0].grid(alpha=.2)
            axes[1].imshow(table.to_numpy()/scale,aspect='auto',cmap='YlOrRd')
            axes[1].set_yticks(range(len(table)),table.index);axes[1].set_xticks(range(self.horizon),table.columns)
            for i in range(len(table)):
                for j in range(self.horizon):axes[1].text(j,i,f'{table.iloc[i,j]/scale:,.0f}',ha='center',va='center',fontsize=8)
            axes[1].set_title(f'{name} ({unit})')
            fig.suptitle(f'{self.target}: horizon-specific specifications; scoring folds={folds or "all"}')
        elif kind=='forecast':
            forecasts=self.forecast(scope);joined=self._joined();bt=joined[joined.task_id.eq('back_test')]
            fig,ax=plt.subplots(figsize=(14,7),layout='constrained')
            count=bt.backtest_id.nunique()
            if backtest_lines is None:backtest_lines=10*count
            if backtest_lines:
                if backtest_lines<0 or backtest_lines%count:raise ValueError('backtest_lines must be a nonnegative multiple of backtest count.')
                top=self.metrics('selection');top=top[top.family.eq('xgb_changes')].groupby('model_id')[self.loss.lower()].mean().sort_values().head(backtest_lines//count).index
                for i,(_,g) in enumerate(bt[bt.model_id.isin(top)].groupby(['model_id','backtest_id'])):
                    g=g.sort_values('horizon');ax.plot(g.month,g.y_hat,color='gray',alpha=.3,lw=.9,label='XGBoost backtests' if i==0 else None)
            ax.plot(self.actuals.index,self.actuals.values,color='black',lw=3,label=f'Observed {self.raw_target}')
            if scope=='global':forecasts=forecasts.assign(display='Best across families')
            else:forecasts=forecasts.assign(display=forecasts.family.map(LABELS))
            for label,g in forecasts.groupby('display',sort=False):
                g=g.sort_values('forecast_month');ax.plot([self.actuals.index[-1],*g.month],[self.actuals.iloc[-1],*g.y_hat],marker='o',lw=2,
                    color=GLOBAL_FORECAST_COLOR if scope=='global' else FAMILY_COLORS[g.family.iloc[0]],label=label)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v/1e6:,.2f}'));ax.set_ylabel('Usage (millions)')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'));ax.legend(fontsize=8);ax.grid(alpha=.2)
            ax.set_title(f'{self.target}: {self.horizon}-month forecast, horizon-specific selection')
        else:raise ValueError('kind must be forecast, mae, or loss.')
        if path:Path(path).parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=170)
        return fig


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--chronos-worker',nargs=2)
    args=parser.parse_args()
    if args.chronos_worker:_chronos_worker(*args.chronos_worker)


xxxx
xxxx
xxxx
xxxx

# %%%
"""Six-month dollar-usage forecasts: one horizon-selected line per family."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUTS = Path('/Users/xavi/Documents/Codex/2026-09-29/acc/outputs')
sys.path.insert(0, str(OUTPUTS))
from forecast_pipeline import MonthlyForecaster

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 850)
pd.set_option('display.max_colwidth', 15)


# %%% main
def main():
    families=[
    "xgb_levels",
    "xgb_changes",
    "random_forest",
    "lasso",
    "chronos_levels",
    "chronos_changes",
    "chronos_finetuned",
    "drift",
    "trend",
    "last_value"]

    targets        =  ["dollar_usage", "chc_usage","dollar_usage_adjusted","chc_usage_adjusted"]
    path_to_data   = ("/Users/xavi/Documents/clickhouse/fcast_one_series/one_series.csv"  )
    path_to_output = ("/Users/xavi/Documents/clickhouse/fcast_one_series/results/main_2")
    scope_         = "global"

    df = pd.read_csv(path_to_data)
    model = MonthlyForecaster(
        df,
        #families=families,
        families=["drift","trend","last_value",'lasso',
                  'xgb_changes',    "chronos_levels",    "chronos_changes"],
        target=["dollar_usage", "dollar_usage_adjusted"],  # Subset
        horizon =17,
        n_backtests=7,
        loss="MAE",
        nonnegative=True,
        output_dir=path_to_output,
    )
    model.train()
    if 0:
        errors = model.metrics()
        best_models = model.select(scope=scope_)
        forecasts = model.forecast(scope=scope_)
        model.save()

        model.plot(
            "loss",
            scope=scope_,
            path=model.output_dir / "loss.png",
        )

        figures = model.plot("forecast", scope=scope_)

        for target, fig in figures.items():
            child = model.models_[target]
            ax = fig.axes[0]

            for line in ax.lines:
                if line.get_label() == "XGBoost backtests":
                    line.set_label(
                        f"XGBoost: 10 models × {child.n_backtests} backtests"
                    )

            ax.axvline(
                child.actuals.index[-1],
                color="black", ls=":", lw=0.9, alpha=0.4,
            )
            ax.set_title(
                f"{target}: horizon-specific {child.horizon}-month forecast"
            )
            ax.set_ylabel(
                "Dollar usage (millions)"
                if target.startswith("dollar_usage")
                else "CHC usage (millions)"
            )
            ax.legend(loc="upper left")

            fig.savefig(
                model.output_dir / f"forecast_{target}.png",
                dpi=180,
            )

        plt.show()
    return model


# %%%
if __name__ == "__main__":
    model = main()
    model.save()

    model.plot("forecast", scope="family")
    model.plot("loss", scope="family")
    model.plot("loss", scope="global")

    fiscal_report = model.fin_report.fy(
        target="dollar_usage_adjusted"
    )
    print(fiscal_report)
    plt.show()
# %%%

  