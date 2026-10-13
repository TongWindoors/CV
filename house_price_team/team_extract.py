"""Re-encode the canonical images with four fixed, frozen FP32 visual towers.

No prices, folds or target-derived features enter this module. The delivered
assets/features are immutable references; fresh encodings live separately in
generated_features. This is feature extraction, not backbone fine-tuning.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request

NAMES = ('register_large336', 'b448', 'l336', 'full')
ATOL, RTOL = 1e-3, 1e-3


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


class PlannedStop(Exception):
    pass


class Guard:
    def __init__(self, root, seconds=None):
        self.root = Path(root)
        self.deadline = time.monotonic() + float(seconds) if seconds else None
    def check(self):
        if any((self.root / name).exists() for name in ('STOP_REQUESTED', 'STOP_REQUESTED.txt')):
            raise PlannedStop('STOP_REQUESTED found; committed work is retained. Remove the stop file before resuming.')
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise PlannedStop('Explicit time budget reached; committed work is retained.')


def sha(path, guard=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            if guard: guard.check()
            h.update(chunk)
    return h.hexdigest()


def resolve(root, value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(root) / p).resolve()


def checked_file(path, expected, guard=None):
    p = Path(path)
    return p.is_file() and p.stat().st_size == expected['bytes'] and sha(p, guard) == expected['sha256']


def download(urls, destination, expected, guard, opener=None):
    """Pinned, verified HTTP download with byte-range resume and mirror fallback."""
    opener = opener or urllib.request.urlopen
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if checked_file(destination, expected, guard): return
        raise ValueError(f'Existing model file fails SHA256; preserved without overwrite: {destination}')
    partial = destination.with_name(destination.name + '.partial')
    if checked_file(partial, expected, guard): partial.replace(destination); return
    if partial.exists() and partial.stat().st_size >= expected['bytes']:
        # Only this downloader-owned incomplete file may be restarted.
        partial.unlink()
    errors = []
    for url in urls:
        for attempt in range(2):
            guard.check()
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'User-Agent': 'HousePrice-Reproduction/1.0', 'Accept-Encoding': 'identity'}
            if offset: headers['Range'] = f'bytes={offset}-'
            try:
                print(f'[下载] {destination.name}，已保存 {offset / 2**20:.1f} MiB；来源 {url.split("/")[2]}', flush=True)
                request = urllib.request.Request(url, headers=headers)
                with opener(request, timeout=30) as response:
                    status = getattr(response, 'status', response.getcode())
                    if status not in (200, 206): raise OSError(f'Unexpected HTTP status {status}')
                    if status == 206:
                        value = response.headers.get('Content-Range', '')
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', value)
                        if not match or int(match[1]) != offset or int(match[3]) != expected['bytes']:
                            raise OSError('Server returned an incompatible byte range; refusing mixed bytes')
                        mode = 'ab' if offset else 'wb'
                    else:
                        # A mirror may ignore Range. Restart our partial stream explicitly.
                        mode, offset = 'wb', 0
                    count, last = offset, time.monotonic()
                    with partial.open(mode) as stream:
                        while True:
                            guard.check(); chunk = response.read(2**20)
                            if not chunk: break
                            stream.write(chunk); count += len(chunk)
                            if count > expected['bytes']: raise OSError('Download exceeds pinned file size')
                            if time.monotonic() - last >= 2 or count == expected['bytes']:
                                print(f'[下载] {destination.name} {count / 2**20:.1f}/{expected["bytes"] / 2**20:.1f} MiB', flush=True)
                                last = time.monotonic()
                if not checked_file(partial, expected, guard):
                    if partial.stat().st_size >= expected['bytes']: partial.unlink()
                    raise OSError('Incomplete download or SHA256 mismatch')
                if destination.exists(): raise FileExistsError('Another process created the destination; refusing overwrite')
                partial.replace(destination)
                return
            except PlannedStop:
                raise
            except (urllib.error.URLError, OSError, ValueError) as error:
                errors.append(f'{url}: {type(error).__name__}: {error}')
                print(f'[下载重试] {str(error)[:180]}', flush=True)
    raise RuntimeError('Pinned model download failed. Partial bytes remain resumable.\n' + '\n'.join(errors))


def ensure_model(root, name, sources, paths, offline, guard):
    spec = sources['models'][name]
    configured = paths.get('models', {}).get(name)
    directory = resolve(root, configured) if configured else Path(root) / 'models_cache' / name / spec['revision']
    if directory.is_file(): directory = directory.parent
    weight = directory / 'model.safetensors'
    if configured:
        if not checked_file(directory / 'config.json', spec['files']['config.json'], guard):
            raise ValueError(f'Configured model config checksum mismatch: {directory}')
        if not any(checked_file(weight, expected, guard) for expected in spec['accepted_local_weights']):
            raise ValueError(f'Configured model weights are not the fixed revision: {weight}')
    else:
        for filename, expected in spec['files'].items():
            if checked_file(directory / filename, expected, guard): continue
            if offline: raise FileNotFoundError(f'Offline mode: missing pinned asset {directory / filename}')
            urls = [f'{host}/{spec["repo"]}/resolve/{spec["revision"]}/{filename}' for host in sources['download_hosts']]
            download(urls, directory / filename, expected, guard)
    return directory, {'repo': spec['repo'], 'revision': spec['revision'], 'weights_sha256': sha(weight, guard),
                       'config_sha256': sha(directory / 'config.json', guard), 'directory': str(directory)}


def verify_input_assets(root, rows_path, reference_root, guard=None):
    """Path overrides may relocate immutable assets, never silently substitute them."""
    manifest = read(Path(root) / 'ASSET_MANIFEST.json', {})
    files = manifest.get('files', {})
    targets = {'data/ROWS.csv': Path(rows_path)}
    targets.update({f'assets/features/{name}.npz': Path(reference_root)/(name+'.npz') for name in NAMES})
    fingerprints = {}
    for key, path in targets.items():
        if key not in files or not checked_file(path, files[key], guard):
            raise ValueError(f'Canonical extraction input checksum mismatch: {key} -> {path}')
        fingerprints[key] = files[key]['sha256']
    return fingerprints


def read_rows(path, probe_count=0):
    import pandas as pd
    # Explicit usecols prevents labels and fold assignment from entering extraction.
    rows = pd.read_csv(path, usecols=['key', 'imageid', 'split', 'image_sha256'], dtype=str)
    if not len(rows) or rows.key.duplicated().any(): raise ValueError('Canonical image keys must be nonempty and unique')
    if not rows['split'].isin(['train', 'test']).all(): raise ValueError('Unknown image split')
    if not (rows.key == rows['split'] + ':' + rows.imageid).all(): raise ValueError('Invalid canonical image key')
    for row in rows.itertuples(index=False):
        if Path(row.imageid).name != row.imageid or '/' in row.imageid or '\\' in row.imageid or row.imageid in ('.', '..'):
            raise ValueError('Image ID must be a simple filename')
        if not re.fullmatch(r'[0-9a-f]{64}', row.image_sha256): raise ValueError('Invalid canonical image SHA256')
    if probe_count:
        import numpy as np
        if probe_count < 1: raise ValueError('Probe count must be positive')
        # Deterministic spread across all inputs, independent of price labels.
        ix = np.unique(np.linspace(0, len(rows)-1, min(probe_count, len(rows)), dtype=int))
        rows = rows.iloc[ix].reset_index(drop=True)
    return rows


def preprocess(image, spec):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(image).convert('RGB')
    size = spec['size']
    if spec['family'] == 'siglip':
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        return (x - .5) / .5, None
    ratio = size / max(image.size)
    width, height = max(1, round(image.width * ratio)), max(1, round(image.height * ratio))
    left, top = (size-width)//2, (size-height)//2
    canvas = Image.new('RGB', (size, size), (124, 116, 104))
    canvas.paste(image.resize((width, height), Image.Resampling.BICUBIC), (left, top))
    x = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255
    x = (x-torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
    mask = torch.zeros(1, size, size); mask[:, top:top+height, left:left+width] = 1
    return x * mask, mask


def readout(output, mask, spec):
    import torch
    from torch.nn import functional as F
    tokens = output.last_hidden_state.float()
    if spec['family'] == 'siglip':
        if tokens.shape[1:] != (256, 1024): raise ValueError('Unexpected SigLIP2 spatial tokens')
        return torch.cat([output.pooler_output.float(), tokens.mean(1)], 1)
    count = (spec['size']//14)**2; registers = spec['register_tokens']
    if tokens.shape[1:] != (1+registers+count, spec['hidden_size']): raise ValueError('Unexpected DINO/register token shape')
    patches = tokens[:, 1+registers:]
    weights = F.avg_pool2d(mask, 14, 14).flatten(2).transpose(1, 2)
    pooled = (patches*weights).sum(1) / weights.sum(1).clamp_min(1e-6)
    return torch.cat([tokens[:, 0], pooled], 1)


def load_model(directory, spec):
    from transformers import AutoModel, SiglipVisionConfig, SiglipVisionModel
    from safetensors import safe_open
    if spec['family'] == 'dino':
        model = AutoModel.from_pretrained(str(directory), local_files_only=True, use_safetensors=True,
                                         trust_remote_code=False, attn_implementation='sdpa')
        if int(model.config.hidden_size) != spec['hidden_size'] or int(getattr(model.config, 'num_register_tokens', 0)) != spec['register_tokens']:
            raise ValueError('DINO configuration does not match locked readout')
    else:
        config = SiglipVisionConfig(**read(directory / 'config.json')['vision_config'])
        config._attn_implementation = 'sdpa'; model = SiglipVisionModel(config)
        # Full official safetensors and the approved local vision-only repack contain
        # the same named visual tensors. Text weights are never instantiated.
        with safe_open(str(directory / 'model.safetensors'), framework='pt', device='cpu') as archive:
            state = {key: archive.get_tensor(key) for key in archive.keys() if key.startswith('vision_model.')}
        model.load_state_dict(state, strict=True); del state
    return model.eval().requires_grad_(False)


def matrix_sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value, dtype='<f4').tobytes()).hexdigest()


def open_cache(directory, signature, count, dim, guard):
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    journal = directory / 'STATE.json'; file = directory / 'features.npy'
    state = read(journal)
    if state:
        if state.get('signature') != signature or not file.is_file(): raise ValueError('Feature-cache binding mismatch')
        values = np.load(file, mmap_mode='r+')
        if values.shape != (count, dim) or values.dtype != np.float32: raise ValueError('Feature-cache shape or precision mismatch')
        done = 0
        for chunk in state['chunks']:
            guard.check()
            a, b = int(chunk['start']), int(chunk['stop'])
            if a != done or not a < b <= count: raise ValueError('Cache chunks are not contiguous and nonoverlapping')
            if not np.isfinite(values[a:b]).all() or matrix_sha(values[a:b]) != chunk['sha256']:
                raise ValueError('Committed feature-cache chunk checksum mismatch')
            done = b
        if state['completed_rows'] != done: raise ValueError('Cache journal row count mismatch')
        return values, state
    values = np.lib.format.open_memmap(file, mode='w+', dtype=np.float32, shape=(count, dim))
    state = {'signature': signature, 'completed_rows': 0, 'chunks': []}; save(journal, state)
    return values, state


def commit_cache(directory, values, state, start, block):
    import numpy as np
    block = np.asarray(block)
    if start != state['completed_rows'] or block.dtype != np.float32 or block.ndim != 2 or block.shape[1] != values.shape[1] or not len(block) or start+len(block)>len(values) or not np.isfinite(block).all():
        raise ValueError('Invalid or overlapping feature-cache commit')
    stop = start + len(block); values[start:stop] = block; values.flush()
    state['chunks'].append({'start': start, 'stop': stop, 'sha256': matrix_sha(block)})
    state['completed_rows'] = stop; save(directory / 'STATE.json', state)


def reference_features(path, rows, dim):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        if set(('keys', 'image_sha256', 'features')) - set(z.files): raise ValueError('Missing reference NPZ fields')
        lookup = {str(key): i for i, key in enumerate(z['keys'])}
        if len(lookup) != len(z['keys']): raise ValueError('Duplicate reference feature keys')
        indices = [lookup[key] for key in rows.key]
        if not np.array_equal(z['image_sha256'][indices], rows.image_sha256.to_numpy()): raise ValueError('Reference feature/image fingerprint mismatch')
        values = z['features'][indices]
    if values.shape != (len(rows), dim) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Reference feature shape, dtype or values are invalid')
    return values


def difference(actual, reference):
    import numpy as np
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Invalid feature comparison')
    delta = a-b; cosine = np.sum(a*b, 1) / np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-12)
    return {'n':len(a), 'max_absolute_difference':float(np.max(abs(delta))), 'rmse':float(np.sqrt(np.mean(delta**2))),
            'mean_cosine_similarity':float(cosine.mean()), 'atol':ATOL, 'rtol':RTOL,
            'within_fp32_tolerance':bool(np.allclose(a, b, atol=ATOL, rtol=RTOL)),
            'interpretation':'Numerical reproduction only; no claim of predictive improvement or backbone training.'}


def export_features(destination, rows, values, signature, audit, guard):
    import numpy as np
    destination = Path(destination); metadata = destination.with_suffix('.json')
    if destination.exists():
        prior = read(metadata, {})
        if prior.get('signature') != signature or prior.get('feature_sha256') != sha(destination, guard):
            raise FileExistsError(f'Existing generated features have a different provenance; preserved: {destination}. Use another --output directory.')
        return prior
    temporary = destination.with_name(destination.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, keys=rows.key.to_numpy(dtype=str), image_sha256=rows.image_sha256.to_numpy(dtype=str), features=np.asarray(values, dtype=np.float32))
    guard.check()
    if destination.exists(): raise FileExistsError('Generated output appeared during extraction; refusing overwrite')
    audit = {**audit, 'signature':signature, 'feature_sha256':sha(temporary, guard), 'n':len(rows), 'status':'complete'}
    # Publish the small metadata first: interruption can leave harmless metadata
    # without the NPZ, but never a completed NPZ without its checksum/provenance.
    save(metadata, audit); guard.check(); temporary.replace(destination)
    return audit


def extract_one(root, output, dataset, feature_root, paths, sources, name, rows, args, guard, hardware):
    import numpy as np
    import torch
    from PIL import Image
    spec = sources['models'][name]
    directory, model_binding = ensure_model(root, name, sources, paths, args.offline, guard)
    reference = reference_features(feature_root / (name+'.npz'), rows, spec['feature_dim'])
    binding = {'format':1, 'model':{k:v for k,v in model_binding.items() if k!='directory'}, 'spec':spec,
               'extractor_sha256':sha(__file__), 'rows':rows.to_dict('records'), 'hardware':hardware, 'precision':'FP32'}
    signature = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    destination = output / (name+'.npz'); metadata = destination.with_suffix('.json')
    if destination.exists():
        prior = read(metadata, {})
        if prior.get('signature') != signature or prior.get('feature_sha256') != sha(destination, guard):
            raise FileExistsError(f'Existing generated features are incompatible and were not overwritten: {destination}')
        values = reference_features(destination, rows, spec['feature_dim'])
        report = difference(values, reference)
        if not report['within_fp32_tolerance']: raise ValueError('Previously generated features fail reproduction tolerance')
        return {**prior, 'status':'reused_complete', 'difference':report}
    cache = output / '.cache' / name / signature
    values, state = open_cache(cache, signature, len(rows), spec['feature_dim'], guard)
    save(cache / 'BINDING.json', binding)
    began = time.monotonic(); resumed = state['completed_rows']
    audit = {'name':name, 'model':model_binding, 'hardware':hardware, 'resumed_rows':resumed,
             'weights_updated':False, 'price_labels_read':False, 'readout':spec['readout'], 'preprocessing':spec['preprocessing']}
    if resumed == len(rows):
        report = difference(values, reference)
        if not report['within_fp32_tolerance']: raise ValueError('Completed cache differs from packaged reference beyond FP32 tolerance')
        return export_features(destination, rows, values, signature, {**audit, 'difference':report}, guard)
    device = torch.device(args.device)
    memory = hardware.get('memory_GiB', 0)
    batch = args.batch if args.batch else (16 if memory >= 20 else 8 if memory >= 10 else 2)
    model = load_model(directory, spec).to(device)
    def encode(indices):
        nonlocal batch
        pieces=[]; offset=0
        while offset < len(indices):
            guard.check(); chosen=indices[offset:offset+batch]
            transformed=[]
            for i in chosen:
                row=rows.iloc[i]; path=dataset/row['split']/row.imageid
                if sha(path, guard)!=row.image_sha256: raise ValueError(f'Image changed during extraction: {row.key}')
                with Image.open(path) as image: transformed.append(preprocess(image, spec))
            x=mask=None
            try:
                x=torch.stack([v[0] for v in transformed]).to(device)
                mask=torch.stack([v[1] for v in transformed]).to(device) if transformed[0][1] is not None else None
                with torch.inference_mode(): result=readout(model(pixel_values=x),mask,spec).cpu().numpy()
            except RuntimeError as error:
                if 'out of memory' not in str(error).lower() or batch<=1: raise
                batch=max(1,batch//2);print(f'[显存调整] {name} batch={batch}，本批未写入缓存，重试。',flush=True)
                del x,mask,transformed;gc.collect()
                if device.type=='cuda':torch.cuda.empty_cache()
                elif device.type=='mps':torch.mps.empty_cache()
                continue
            if result.shape!=(len(chosen),spec['feature_dim']) or not np.isfinite(result).all(): raise ValueError('Unexpected extracted features')
            pieces.append(result.astype(np.float32,copy=False));offset+=len(chosen)
        return np.concatenate(pieces)
    try:
        probes=np.unique(np.linspace(0,len(rows)-1,min(2,len(rows)),dtype=int)).tolist()
        probe_difference=difference(encode(probes),reference[probes])
        save(output/(name+'_PROBE.json'),{'keys':rows.key.iloc[probes].tolist(),**probe_difference})
        if not probe_difference['within_fp32_tolerance']:
            raise ValueError(f'{name} probe differs from locked reference; see {name}_PROBE.json. Stop to check preprocessing/runtime before full extraction.')
        while state['completed_rows'] < len(rows):
            guard.check(); start=state['completed_rows']; stop=min(len(rows),start+batch)
            block=encode(list(range(start,stop)));commit_cache(cache,values,state,start,block)
            elapsed=time.monotonic()-began;done=state['completed_rows'];rate=(done-resumed)/max(elapsed,.01)
            print(f'[提取] {name} {done}/{len(rows)} | {rate:.2f} 张/秒 | batch={batch}',flush=True)
            save(output/'EXTRACTION_PROGRESS.json',{'branch':name,'done':done,'total':len(rows),'batch':batch,'elapsed_seconds':elapsed,'cache':str(cache)})
        report=difference(values,reference)
        save(output/(name+'_DIFFERENCE.json'),report)
        if not report['within_fp32_tolerance']:raise ValueError(f'{name} generated features exceed FP32 reproduction tolerance; retained cache and difference report for diagnosis')
        audit.update(difference=report,elapsed_seconds=time.monotonic()-began,batch=batch)
        return export_features(destination,rows,values,signature,audit,guard)
    finally:
        del model;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
        elif device.type=='mps':torch.mps.empty_cache()


def validate_generated_features(root, folder, rows):
    """Strict gate for --features generated output, never for default cached assets.

    Return auditable branch fingerprints after independently recomputing numerical
    differences. Merely copying NPZ files or a success flag cannot pass this gate.
    """
    import numpy as np
    root, folder = Path(root).resolve(), Path(folder).resolve()
    source_file = root / 'MODEL_SOURCES.json'
    sources = read(source_file)
    status = read(folder / 'EXTRACTION_STATUS.json', {})
    if status.get('status') != 'complete' or status.get('probe_only') is not False:
        raise ValueError('Generated feature set is incomplete or a probe; finish all four branches first')
    if set(status.get('branches', {})) != set(NAMES):
        raise ValueError('Generated feature set must contain all four completed branches')
    if status.get('backbones_finetuned') is not False or status.get('prices_read') is not False:
        raise ValueError('Generated feature extraction protocol flags are invalid')
    extractor_sha = sha(root / 'team_extract.py')
    if status.get('sources_sha256') != sha(source_file) or status.get('extractor_sha256') != extractor_sha:
        raise ValueError('Generated features do not match the fixed model sources/extractor')
    canonical = rows[['key', 'imageid', 'split', 'image_sha256']].astype(str).reset_index(drop=True)
    if len(canonical) != 11000 or canonical.key.duplicated().any():
        raise ValueError('Expected all 11000 canonical image rows for generated features')
    if not (canonical.key == canonical['split'] + ':' + canonical.imageid).all():
        raise ValueError('Canonical row keys do not match image splits')
    paths = read(root / 'paths.json', {})
    reference_root = resolve(root, paths.get('features', paths.get('feature_dir', 'assets/features')))
    if reference_root == folder:
        raise ValueError('The immutable reference cannot be the generated feature directory itself')
    rows_path=resolve(root,paths.get('rows','data/ROWS.csv'))
    input_sha=verify_input_assets(root,rows_path,reference_root)
    locked_rows=read_rows(rows_path)[['key','imageid','split','image_sha256']].astype(str).reset_index(drop=True)
    if not canonical.equals(locked_rows):
        raise ValueError('Generated feature request differs from locked canonical image rows')
    result = {'status':'passed', 'rows':len(canonical), 'generated_root':str(folder),
              'extractor_sha256':extractor_sha, 'sources_sha256':sha(source_file), 'branches':{},
              'rows_identity_sha256':hashlib.sha256(json.dumps(canonical.to_dict('records'),sort_keys=True).encode()).hexdigest(),
              'claim':'Frozen numerical reproduction only; no GPU predictive-gain claim'}
    result['canonical_input_sha256']=input_sha
    for name in NAMES:
        spec = sources['models'][name]; metadata = read(folder/(name+'.json'), {})
        binding = metadata.get('model', {}); hardware = metadata.get('hardware', {})
        if metadata.get('status') != 'complete' or metadata.get('n') != len(canonical):
            raise ValueError(f'{name}: generated metadata is not a complete full-data extraction')
        if metadata.get('weights_updated') is not False or metadata.get('price_labels_read') is not False:
            raise ValueError(f'{name}: invalid extraction training/label-use flags')
        if binding.get('repo') != spec['repo'] or binding.get('revision') != spec['revision']:
            raise ValueError(f'{name}: model repository/revision differs from locked recipe')
        if binding.get('weights_sha256') not in {x['sha256'] for x in spec['accepted_local_weights']} or binding.get('config_sha256') != spec['files']['config.json']['sha256']:
            raise ValueError(f'{name}: model/config fingerprint is not approved')
        if metadata.get('preprocessing') != spec['preprocessing'] or metadata.get('readout') != spec['readout']:
            raise ValueError(f'{name}: preprocessing/readout differs from the original representation')
        if hardware.get('device') != 'cuda' or hardware.get('precision') != 'FP32':
            raise ValueError(f'{name}: full production regeneration requires CUDA FP32')
        protocol = {'format':1, 'model':{k:v for k,v in binding.items() if k!='directory'}, 'spec':spec,
                    'extractor_sha256':extractor_sha, 'rows':canonical.to_dict('records'),
                    'hardware':hardware, 'precision':'FP32'}
        expected_signature = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        if metadata.get('signature') != expected_signature:
            raise ValueError(f'{name}: feature binding does not match canonical rows/model/preprocessing/runtime')
        path = folder/(name+'.npz'); digest=sha(path)
        if metadata.get('feature_sha256') != digest or status['branches'][name].get('feature_sha256') != digest:
            raise ValueError(f'{name}: generated NPZ checksum differs from completion records')
        with np.load(path,allow_pickle=False) as archive:
            if not np.array_equal(archive['keys'],canonical.key.to_numpy()) or not np.array_equal(archive['image_sha256'],canonical.image_sha256.to_numpy()):
                raise ValueError(f'{name}: generated NPZ rows/order differ from canonical inputs')
        actual = reference_features(path, canonical, spec['feature_dim'])
        reference_path=reference_root/(name+'.npz')
        reference=reference_features(reference_path, canonical, spec['feature_dim'])
        report=difference(actual,reference)
        if not metadata.get('difference',{}).get('within_fp32_tolerance') or not report['within_fp32_tolerance']:
            raise ValueError(f'{name}: GPU/reference differences exceed the fixed FP32 tolerance')
        result['branches'][name]={'feature_sha256':digest, 'reference_sha256':sha(reference_path),
                                  'signature':expected_signature, 'model':binding, 'difference':report}
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--paths',help='Optional paths.json outside package for local asset overrides')
    parser.add_argument('--output',help='Separate output directory; default root/generated_features')
    parser.add_argument('--names',nargs='+',choices=NAMES,default=list(NAMES))
    parser.add_argument('--device',choices=('cuda','cpu','mps'),default='cuda')
    parser.add_argument('--probe-count',type=int,default=0,help='Small mechanical check only; no full reproduction claim')
    parser.add_argument('--batch',type=int,default=0)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--max-seconds',type=float,default=None,help='Optional explicit budget; no default time limit')
    args=parser.parse_args(argv);root=Path(args.root).resolve()
    output=resolve(root,args.output) if args.output else root/('probe_features' if args.probe_count else 'generated_features')
    output.mkdir(parents=True,exist_ok=True)
    report={'status':'running','branches':{},'backbones_finetuned':False,'prices_read':False,'probe_only':bool(args.probe_count),'started_unix':time.time()}
    try:
        if args.batch<0 or args.probe_count<0 or (args.max_seconds is not None and args.max_seconds<=0):raise ValueError('Invalid batch/probe/time value')
        if args.device!='cuda' and not 1<=args.probe_count<=8:raise ValueError('Non-CUDA devices are restricted to an explicit --probe-count 1..8 mechanical check')
        if args.probe_count and output.name=='generated_features':raise ValueError('Probe output must use a separate directory, not generated_features')
        paths=read(resolve(root,args.paths) if args.paths else root/'paths.json',{})
        dataset=resolve(root,paths.get('dataset',paths.get('dataset_dir','assets/dataset')))
        feature_root=resolve(root,paths.get('features',paths.get('feature_dir','assets/features')))
        if output==feature_root:raise ValueError('Output must differ from immutable reference features')
        sources=read(root/'MODEL_SOURCES.json');guard=Guard(root,args.max_seconds);guard.check()
        import torch
        rows_path=resolve(root,paths.get('rows','data/ROWS.csv'))
        print('[资源校验] 核对固定canonical rows与四组原始参考特征',flush=True)
        input_sha=verify_input_assets(root,rows_path,feature_root,guard)
        report['canonical_input_sha256']=input_sha
        rows=read_rows(rows_path,args.probe_count)
        torch.set_num_threads(2);torch.manual_seed(17);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if args.device=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('NVIDIA CUDA is unavailable. Install the matching CUDA PyTorch environment and NVIDIA driver; cached CPU reproduction does not require GPU.')
            info=torch.cuda.get_device_properties(0)
            hardware={'device':'cuda','name':info.name,'memory_GiB':info.total_memory/2**30,'capability':list(torch.cuda.get_device_capability(0))}
        elif args.device=='mps':
            if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
            hardware={'device':'mps','name':'local MPS mechanical probe'}
        else:hardware={'device':'cpu','name':'CPU mechanical probe'}
        hardware.update(torch=torch.__version__,cuda=torch.version.cuda,precision='FP32',versions={p:importlib.metadata.version(p) for p in ('torchvision','transformers','safetensors','pillow','numpy')})
        report.update(hardware=hardware,rows=len(rows),sources_sha256=sha(root/'MODEL_SOURCES.json'),extractor_sha256=sha(__file__))
        print(f'[设备] {hardware["name"]} | {len(rows)} 张 | FP32冻结提取 | 不读取房价、不更新视觉模型权重',flush=True)
        print('[图片校验] 校验所有原始图片 SHA256，包括将要复用的缓存行。',flush=True)
        for i,row in enumerate(rows.itertuples(index=False)):
            guard.check()
            if sha(dataset/row.split/row.imageid,guard)!=row.image_sha256:raise ValueError(f'Image checksum mismatch: {row.key}')
            if (i+1)%1000==0 or i+1==len(rows):print(f'[图片校验] {i+1}/{len(rows)}',flush=True)
        for name in dict.fromkeys(args.names):
            guard.check();print(f'[模型] {name}',flush=True)
            report['current_branch']=name;save(output/'EXTRACTION_STATUS.json',report)
            report['branches'][name]=extract_one(root,output,dataset,feature_root,paths,sources,name,rows,args,guard,hardware)
            save(output/'GPU_DIFFERENCE_REPORT.json',report)
        report['status']='probe_complete' if args.probe_count else 'complete'
        report['scope']='Feature numerical reproduction; re-run identical regression recipes to compare MSE. GPU extraction is not fine-tuning.'
        save(output/'GPU_DIFFERENCE_REPORT.json',report);save(output/'EXTRACTION_STATUS.json',report)
        print(f'[完成] 特征保存在 {output}；数值差异见 GPU_DIFFERENCE_REPORT.json。',flush=True)
        return 0
    except (PlannedStop,KeyboardInterrupt) as error:
        report.update(status='interrupted',error=str(error),incomplete=True)
        save(output/'GPU_DIFFERENCE_REPORT.json',report);save(output/'EXTRACTION_STATUS.json',report)
        print(f'[停止] {error}',flush=True);return 2
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),incomplete=True)
        save(output/'GPU_DIFFERENCE_REPORT.json',report);save(output/'EXTRACTION_STATUS.json',report)
        print(f'[失败] {error}',flush=True);traceback.print_exc();return 1


if __name__=='__main__':sys.exit(main())
