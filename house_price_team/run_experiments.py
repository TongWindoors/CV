"""Portable, matched Baseline/V1/V2/V3 reproduction. No validation-driven search."""
from pathlib import Path
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import argparse
import datetime as dt
import hashlib
import html
import json
import platform
import shutil
import sys
import time
import traceback
import zipfile
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import team_core as core

ROOT = Path(__file__).resolve().parent
NAMES = ('register_large336', 'b448', 'l336', 'full')
DIMS = dict(zip(NAMES, (2048, 1536, 2048, 2048)))
EXPECTED_COUNTS = {'inner_train':5118, 'inner_calibration':1280,
                   'historical_development':1599, 'excluded_near_development':3, 'test':3000}


class Tee:
    def __init__(self, terminal, log): self.terminal,self.log=terminal,log
    def write(self, value):
        self.terminal.write(value); self.log.write(value); self.log.flush()
        return len(value)
    def flush(self): self.terminal.flush(); self.log.flush()
    def isatty(self): return False


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def check_stop(root):
    if (root/'STOP_REQUESTED').exists():
        raise KeyboardInterrupt('STOP_REQUESTED: completed files will be retained.')


def validate_rows(rows, train, test, strict=True):
    required = {'key','imageid','split','price','image_sha256','group','partition','outer_fold'}
    if not required.issubset(rows): raise ValueError('Missing canonical row columns')
    if not rows.key.is_unique or rows.key.isna().any(): raise ValueError('Duplicate/missing canonical key')
    if not np.array_equal(rows.key, rows.split + ':' + rows.imageid): raise ValueError('Key/order mapping changed')
    if any('/' in str(x) or '\\' in str(x) or ':' in str(x) or str(x) in ('.','..') for x in rows.imageid):
        raise ValueError('Unsafe image filename')
    tr=rows.loc[rows.split.eq('train')]; te=rows.loc[rows.split.eq('test')]
    if not tr.imageid.is_unique or not te.imageid.is_unique: raise ValueError('Duplicate image IDs')
    if not np.array_equal(tr.imageid,train.imageid) or not np.array_equal(te.imageid,test.imageid):
        raise ValueError('Original train/test row order changed')
    if not np.isfinite(tr.price).all() or not te.price.isna().all(): raise ValueError('Invalid labels / test labels present')
    if not np.allclose(tr.price,train.price,atol=1e-9,rtol=0): raise ValueError('Training labels changed')
    if strict and rows.partition.value_counts().to_dict()!=EXPECTED_COUNTS: raise ValueError('Partition counts changed')
    if not rows.image_sha256.astype(str).str.fullmatch('[a-f0-9]{64}').all(): raise ValueError('Invalid image SHA')
    if not te.partition.eq('test').all() or tr.partition.eq('test').any(): raise ValueError('Wrong train/test partition')
    pool=rows.partition.eq('inner_train')
    if set(rows.loc[pool,'outer_fold'])!={0,1,2} or not rows.loc[~pool,'outer_fold'].eq(-1).all():
        raise ValueError('Invalid original three-fold plan')
    if strict and rows.loc[pool,'outer_fold'].value_counts().to_dict()!={0:1706,1:1706,2:1706}:
        raise ValueError('Original fold membership/counts changed')
    if rows.loc[pool].groupby('group').outer_fold.nunique().max()!=1: raise ValueError('Duplicate group crosses folds')
    parts=['inner_train','inner_calibration','historical_development']
    for i,a in enumerate(parts):
        for b in parts[i+1:]:
            if set(rows.loc[rows.partition.eq(a),'group']) & set(rows.loc[rows.partition.eq(b),'group']):
                raise ValueError('Duplicate group crosses historical partitions')


def load(root, feature_dir=None):
    manifest=read(root/'ASSET_MANIFEST.json')
    for relative in ['data/ROWS.csv','data/train.csv','data/test.csv']:
        if sha(root/relative)!=manifest['files'][relative]['sha256']: raise ValueError(f'Canonical data hash mismatch: {relative}')
    rows=pd.read_csv(root/'data/ROWS.csv',float_precision='round_trip')
    validate_rows(rows,pd.read_csv(root/'data/train.csv',float_precision='round_trip'),pd.read_csv(root/'data/test.csv'))
    folder=Path(feature_dir).resolve() if feature_dir else root/'assets/features'
    raw={}; bindings={}
    if feature_dir:
        from team_extract import validate_generated_features
        bindings['_regenerated_validation']=validate_generated_features(root,folder,rows)
    for name in NAMES:
        check_stop(root); file=folder/f'{name}.npz'
        if not file.is_file(): raise FileNotFoundError(f'Missing {file}. Download HOUSE_PRICE_TEAM_FULL.zip from the project Release and extract it completely.')
        digest=sha(file)
        if not feature_dir and digest!=manifest['files'][f'assets/features/{name}.npz']['sha256']:
            raise ValueError(f'Cached feature hash mismatch: {name}')
        with np.load(file,allow_pickle=False) as z:
            if not np.array_equal(z['keys'],rows.key) or not np.array_equal(z['image_sha256'],rows.image_sha256):
                raise ValueError(f'Feature keys/image fingerprints changed: {name}')
            x=z['features'].copy()
        if x.shape!=(len(rows),DIMS[name]) or x.dtype!=np.float32 or not np.isfinite(x).all():
            raise ValueError(f'Invalid FP32 feature matrix: {name}')
        raw[name]=x; bindings[name]={'sha256':digest,'bytes':file.stat().st_size,'shape':list(x.shape)}
        print(f'[data] {name}: {x.shape}; SHA256 verified',flush=True)
    return rows,raw,bindings


def predict_fit(raw, y, fit_idx, eval_idx, spec):
    """The only supervised inputs are y[fit_idx]; test/CAL/OOF labels never fit a model."""
    if np.intersect1d(fit_idx,eval_idx).size: raise ValueError('Supervised fit/assessment overlap')
    labels=np.asarray(y[fit_idx],dtype=np.float64)
    if not np.isfinite(labels).all(): raise ValueError('Nonfinite fit labels')
    needed=list(core.expanded_weights(spec['weights']))
    tr={n:core.normalize(raw[n][fit_idx]) for n in needed}
    ev={n:core.normalize(raw[n][eval_idx]) for n in needed}
    if spec['regressor']=='ridge':
        mean=float(labels.mean()); scale=max(1.,float(labels.std()))
        model=Ridge(alpha=spec['alpha'],solver='cholesky').fit(tr['register_large336'],(labels-mean)/scale)
        pred=np.maximum(model.predict(ev['register_large336'])*scale+mean,1.)
        state={'coef':model.coef_,'intercept':np.asarray(model.intercept_), 'target_mean':np.asarray(mean),'target_scale':np.asarray(scale)}
    else:
        model=core.fit(tr,labels,spec['weights'],gamma=spec['gamma'],alpha=spec['alpha'])
        pred=core.predict(model,tr,ev)
        state={k:np.asarray(v) for k,v in model.items() if k not in ('weights',)}
    if pred.shape!=(len(eval_idx),) or not np.isfinite(pred).all(): raise ValueError('Invalid predictions')
    return pred,state


def prediction_frame(rows,indices,pred,method,fold):
    df=rows.iloc[indices][['key','imageid','image_sha256','group','price']].copy()
    df['prediction']=pred; df['method']=method; df['fold']=fold
    return df.rename(columns={'price':'actual'})


def descriptive_intervals(predictions, methods, samples=2000):
    base=predictions[predictions.method.eq('baseline')].sort_values('key')
    if base.empty: return []
    out=[]
    for name in methods:
        if name=='baseline': continue
        candidate=predictions[predictions.method.eq(name)].sort_values('key')
        if not np.array_equal(base.key,candidate.key): raise ValueError('Paired comparison keys differ')
        delta=(candidate.prediction.to_numpy()-candidate.actual.to_numpy())**2-(base.prediction.to_numpy()-base.actual.to_numpy())**2
        frame=pd.DataFrame({'group':base.group.to_numpy(),'d':delta}).groupby('group').d.agg(['sum','count'])
        sums=frame['sum'].to_numpy();counts=frame['count'].to_numpy();rng=np.random.default_rng(20261013)
        draws=[]
        for _ in range(samples):
            idx=rng.integers(0,len(sums),len(sums));draws.append(float(sums[idx].sum()/counts[idx].sum()))
        low,high=np.quantile(draws,[.025,.975])
        out.append({'comparison':f'{name} minus baseline','delta_mse':float(delta.mean()),'descriptive_95pct':[float(low),float(high)],'bootstrap_groups':len(sums),'draws':samples,
                    'scope':'Conditional on historical predictions; excludes retraining/model-selection uncertainty; negative favors candidate.'})
    return out


def html_report(out,scores,folds,status,intervals):
    records=scores.to_dict('records') if len(scores) else []
    bars=''
    maximum=max([r['mse'] for r in records]+[1.])
    for i,r in enumerate(records):
        y=25+i*55; width=440*r['mse']/maximum
        bars+=f'<text x="8" y="{y+18}">{html.escape(r["method"])}</text><rect x="110" y="{y}" width="{width:.1f}" height="30" fill="#316aad"/><text x="{120+width:.1f}" y="{y+20}">{r["mse"]:,.2f}</text>'
    visual=f'<svg viewBox="0 0 700 {max(80,len(records)*55+35)}" role="img" aria-label="历史三折 MSE 比较">{bars}</svg>'
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>四版本复现报告</title>
<style>body{{max-width:1050px;margin:40px auto;padding:0 22px;font:17px/1.7 system-ui;color:#152335}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;font-size:14px;display:block;overflow:auto}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}svg{{max-width:850px;width:100%}}.note{{background:#edf3fa;padding:16px}}pre{{white-space:pre-wrap}}</style>
<h1>Baseline → V1 → V2 → V3</h1><p>状态：{html.escape(status['status'])}。来源：{html.escape(status.get('feature_source',''))}。</p>
<p class="note">主比较使用同一 5,118 张历史训练池、同一三个分组折；每折拟合 3,412 张，评估 1,706 张。历史数据与方法选择均曾被使用，因此这是复现和探索性证据，不能称为新盲测，也不是 Kaggle 分数。</p>
<p>Baseline：冻结 DINO + Ridge；V1：同一特征 + RBF KRR；V2：三个冻结 DINO 表征 + 同一 KRR；V3：再融合 SigLIP2 视觉表征。所有视觉模型权重冻结；回归参数使用本轮训练标签重新拟合。</p>
{visual}<h2>汇总</h2>{scores.to_html(index=False,float_format=lambda x:f'{x:,.4f}') if len(scores) else '<p>暂无完整版本。</p>'}
<h2>逐折结果</h2>{folds.to_html(index=False,float_format=lambda x:f'{x:,.4f}') if len(folds) else '<p>暂无完整折。</p>'}
<h2>配对描述性区间</h2><pre>{html.escape(json.dumps(intervals,ensure_ascii=False,indent=2))}</pre>
<h2>解释边界与下一步</h2><p>V1→V2 测试多模型表征融合，不能单独归因于分辨率、参数量或某一个骨干。V2→V3 测试加入不同预训练目标的视觉表征。差异不能归因于仅仅用了 GPU；GPU 用于重新编码原图，CPU 精确求解回归权重。</p>
<p>如存在 submissions/，其中每个 CSV 使用全部 8,000 个标签另行重拟合后预测 3,000 张测试图片。上面的 OOF 分数评估的是每折模型，不是全量重拟合模型。保留 Kaggle 实测分数另列报告，不混写。</p>
<p>参考 evidence/ 的历史微调结果说明已经做过冻结头、MLP、末四层和 LoRA 的比较；尚未返回结果的全参数微调不写成成功实验。课程对 SigLIP2 双编码器的最终界定仍以老师确认为准，V2 提供纯视觉路线。</p>
</html>'''
    (out/'REPORT.html').write_text(text,encoding='utf-8')


def collect(root,out):
    target=root/'TEAM_RESULTS.zip'; temporary=root/'TEAM_RESULTS.zip.tmp'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.rglob('*')):
            if not file.is_file() or 'models' in file.relative_to(out).parts: continue
            z.write(file,'run/'+file.relative_to(out).as_posix())
        for name in ['METHODS.json','data/PROVENANCE.json','MODEL_SOURCES.json']:
            if (root/name).is_file(): z.write(root/name,'context/'+name)
        logfiles=sorted(f for f in (root/'logs').glob('*') if f.suffix in ('.log','.txt'))
        for file in logfiles[-8:]:
            # Installation log can still be open; completed bytes are diagnostic only.
            try: z.write(file,'launcher_logs/'+file.name)
            except OSError: pass
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None: raise IOError('Result archive failed CRC verification')
    temporary.replace(target)
    print(f'[results] {target}',flush=True)
    return target


def collect_extraction(root):
    """GPU diagnostics stay small: no tensors, weights, images or environments."""
    target=root/'GPU_REEXTRACT_RESULTS.zip';temporary=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted((root/'generated_features').glob('*.json')):
            z.write(file,'generated_features/'+file.name)
        for name in ['team_extract.py','MODEL_SOURCES.json','ASSET_MANIFEST.json']:
            z.write(root/name,'source/'+name)
        for file in sorted(f for f in (root/'logs').glob('*') if f.suffix in ('.log','.txt'))[-8:]:
            z.write(file,'logs/'+file.name)
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None: raise IOError('GPU diagnostic archive failed CRC')
    temporary.replace(target)
    print(f'[GPU diagnostics] {target}',flush=True)


def run(args):
    root=Path(args.root).resolve()
    recipe=read(root/'METHODS.json'); selected=list(recipe['methods']) if args.method=='all' else [args.method]
    if any(n not in recipe['methods'] for n in selected): raise ValueError('Unknown method')
    stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+os.urandom(3).hex()
    out=root/'runs'/stamp;out.mkdir(parents=True)
    if (root/'TEAM_RESULTS.zip').is_file():
        (root/'result_history').mkdir(exist_ok=True)
        (root/'TEAM_RESULTS.zip').replace(root/'result_history'/f'TEAM_RESULTS_before_{stamp}.zip')
    save(root/'LATEST_RUN.json',{'run':'runs/'+stamp})
    for sub in ['source_snapshot','predictions','models','submissions']: (out/sub).mkdir()
    for file in root.iterdir():
        if file.is_file() and file.suffix in ('.py','.json','.ps1','.cmd','.txt'):
            shutil.copy2(file,out/'source_snapshot'/file.name)
    for name in ['ROWS.csv','PROVENANCE.json']:
        shutil.copy2(root/'data'/name,out/name)
    status={'status':'running','started_at':dt.datetime.now().isoformat(),'feature_source':'regenerated' if args.features else 'shipped_frozen_cache','methods':selected,
            'scope':'Matched historical OOF reproduction; not a new blind test or Kaggle score.', 'gpu_finetuning':False,'python':sys.version,'platform':platform.platform()}
    save(out/'STATUS.json',status)
    folds=[];predictions=[];scores=[];intervals=[];began=time.monotonic();exit_code=0
    try:
        rows,raw,binding=load(root,args.features); save(out/'INPUT_BINDING.json',binding)
        y=rows.price.to_numpy(dtype=np.float64);pool=rows.partition.to_numpy()=='inner_train'
        fold_id=rows.outer_fold.to_numpy()
        mean_frames=[]
        for f in (0,1,2):
            fit=np.flatnonzero(pool&(fold_id!=f)); val=np.flatnonzero(pool&(fold_id==f))
            mean_frames.append(prediction_frame(rows,val,np.full(len(val),y[fit].mean()),'mean_sanity',f))
        means=pd.concat(mean_frames,ignore_index=True)
        save(out/'MEAN_SANITY.json',core.metrics(means.actual,means.prediction))
        for name in selected:
            spec=recipe['methods'][name]; branch=[]
            for f in (0,1,2):
                check_stop(root);fit=np.flatnonzero(pool&(fold_id!=f));val=np.flatnonzero(pool&(fold_id==f))
                if set(rows.iloc[fit].group)&set(rows.iloc[val].group): raise ValueError('Group leakage')
                print(f'[{name}] fold {f+1}/3: fit={len(fit)}, validate={len(val)}; fitting {spec["regressor"]}...',flush=True)
                start=time.monotonic();p,_=predict_fit(raw,y,fit,val,spec)
                record={'method':name,'fold':f,'n_fit':len(fit),'n_validation':len(val),**core.metrics(y[val],p),'seconds':time.monotonic()-start}
                folds.append(record);frame=prediction_frame(rows,val,p,name,f);branch.append(frame);predictions.append(frame)
                frame.to_csv(out/'predictions'/f'{name}_fold{f}.csv',index=False)
                pd.DataFrame(folds).to_csv(out/'SCORES_BY_FOLD.csv',index=False)
                print(f'[{name}] fold {f+1}: MSE={record["mse"]:.4f}, MAE={record["mae"]:.4f}, {record["seconds"]:.1f}s',flush=True)
            combined=pd.concat(branch,ignore_index=True)
            if combined.key.nunique()!=5118: raise ValueError('Incomplete OOF coverage')
            score={'method':name,'label':spec['label'],'n':len(combined),**core.metrics(combined.actual,combined.prediction)}
            scores.append(score);pd.DataFrame(scores).to_csv(out/'SCORES.csv',index=False)
            print(f'[{name}] OOF MSE={score["mse"]:.4f}',flush=True)
        allpred=pd.concat(predictions,ignore_index=True).sort_values(['method','key'])
        allpred.to_csv(out/'OOF_PREDICTIONS.csv',index=False)
        intervals=descriptive_intervals(allpred,selected);save(out/'PAIRED_INTERVALS.json',intervals)
        # A mean baseline verifies that using labels has real predictive value.
        # Historical parity is diagnostic; regenerated GPU features may have small numerical drift.
        historical=read(root/'evidence/EXPECTED_SCORES.json')
        parity=[{'method':s['method'],'expected_historical_mse':historical['mse'][s['method']], 'current_mse':s['mse'],
                 'absolute_difference':abs(s['mse']-historical['mse'][s['method']]),'within_0_1':bool(abs(s['mse']-historical['mse'][s['method']])<=.1)} for s in scores if s['method'] in historical['mse']]
        save(out/'PARITY.json',{'comparisons':parity,'identical_features_expected':not bool(args.features),'not_gpu_accuracy_validation':True})
        if not args.features and not all(r['within_0_1'] for r in parity):
            raise ValueError('Historical cache reproduction differed by >0.1 MSE. Evidence saved; no final refit will be published.')
        if not args.no_refit:
            fit=np.flatnonzero(rows.split.to_numpy()=='train');test=np.flatnonzero(rows.split.to_numpy()=='test')
            for name in selected:
                check_stop(root);spec=recipe['methods'][name]
                print(f'[{name}] full refit: 8000 labels -> 3000 test predictions...',flush=True)
                p,state=predict_fit(raw,y,fit,test,spec)
                submission=pd.DataFrame({'imageid':rows.iloc[test].imageid.to_numpy(),'price':p})
                if list(submission.columns)!=['imageid','price'] or len(submission)!=3000 or not np.isfinite(p).all() or (p<=0).any():
                    raise ValueError('Submission schema/values incorrect')
                state.update(training_keys=rows.iloc[fit].key.to_numpy(dtype=str),metadata=np.asarray(json.dumps({'method':name,'recipe':spec,'binding':binding,'normalization':'two half row-L2 blocks / sqrt(2), float64','lower_clip':1.})))
                with (out/'models'/f'{name}.npz').open('wb') as stream: np.savez_compressed(stream,**state)
                submission.to_csv(out/'submissions'/f'SUBMISSION_{name}.csv',index=False)
                print(f'[{name}] submission saved. OOF scores do not evaluate this all-label refit.',flush=True)
        status.update(status='exporting',computation_complete=True,full_refit_completed=not args.no_refit)
    except KeyboardInterrupt as error:
        status.update(status='interrupted',error=str(error));exit_code=2
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}');exit_code=1
        (out/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8');traceback.print_exc()
    finally:
        status['elapsed_seconds']=time.monotonic()-began;save(out/'STATUS.json',status)
        try:
            if predictions: pd.concat(predictions,ignore_index=True).to_csv(out/'PARTIAL_PREDICTIONS.csv',index=False)
            table=pd.DataFrame(scores);foldtable=pd.DataFrame(folds)
            if status['status']=='exporting': status['status']='complete'
            html_report(out,table,foldtable,status,intervals)
            summary='状态：'+status['status']+'\n'+'特征缓存复现；回归权重重新训练；没有视觉骨干微调。\n'
            summary+='同一历史 5118 张/3 折；不是 Kaggle 分数。\n'
            summary+='\n'.join(f'{s["method"]}: MSE={s["mse"]:.4f}; MAE={s["mae"]:.4f}' for s in scores)
            summary+='\n全量重拟合完成：'+str(status.get('full_refit_completed',False))+'\n'
            if status.get('error'): summary+=status['error']+'\n'
            (out/'RESULT_SUMMARY.txt').write_text(summary,encoding='utf-8')
            save(out/'STATUS.json',status)
            collect(root,out)
            print(summary,flush=True)
        except (Exception,KeyboardInterrupt) as error:
            status.update(status='delivery_failed',delivery_error=f'{type(error).__name__}: {error}')
            exit_code=1
            # Never leave a failed current run represented by the previous run's ZIP.
            for target in [root/'TEAM_RESULTS.zip',root/'TEAM_RESULTS.zip.tmp']:
                try: target.unlink(missing_ok=True)
                except OSError: pass
            try:
                save(out/'STATUS.json',status)
                (out/'DELIVERY_ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
                (out/'RESULT_SUMMARY.txt').write_text('状态：delivery_failed\n'+str(error),encoding='utf-8')
                collect(root,out)
            except Exception: pass
            print(f'[FAILED] Result export failed: {error}. See {out}',flush=True)
    return exit_code


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default=str(ROOT));ap.add_argument('--mode',choices=['all','extract','collect'],default='all')
    ap.add_argument('--method',choices=['all','baseline','v1','v2','v3'],default='all')
    ap.add_argument('--features',help='Explicit regenerated feature directory; never silently substitutes historical caches')
    ap.add_argument('--no-refit',action='store_true',help='OOF evidence only; do not fit 8000-label final models')
    args=ap.parse_args(argv);root=Path(args.root).resolve()
    if args.mode=='collect':
        collect(root,root/read(root/'LATEST_RUN.json')['run']);return 0
    if (root/'STOP_REQUESTED').exists():
        print('Remove STOP_REQUESTED when you are ready to start a NEW run.',flush=True);return 2
    lock=root/'.run.lock'
    try:
        descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        print('Another run holds .run.lock. If ALL run windows are closed, remove this stale lock and retry.',flush=True);return 1
    os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    try:
        (root/'logs').mkdir(exist_ok=True)
        logpath=root/'logs'/('run_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())+'.log')
        with logpath.open('w',encoding='utf-8',buffering=1) as log:
            previous_stdout,previous_stderr=sys.stdout,sys.stderr
            sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
            try:
                if args.mode=='extract':
                    from team_extract import main as extract_main
                    try: return extract_main(['--root',str(root)])
                    finally: collect_extraction(root)
                with threadpool_limits(limits=2): return run(args)
            finally: sys.stdout,sys.stderr=previous_stdout,previous_stderr
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    raise SystemExit(main())
