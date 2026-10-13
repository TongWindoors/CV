"""Fixed centered-RBF kernel ridge fitting; no hyperparameter search or hidden selection."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

DINO_WEIGHTS={'register_large336':.4,'b448':.15,'l336':.45}

def normalize(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]%2 or not np.isfinite(x).all():
        raise ValueError('Expected finite two-block visual features')
    return np.concatenate([a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)/np.sqrt(2)
                           for a in np.split(x,2,axis=1)],axis=1)

def expanded_weights(weights):
    if not np.isclose(sum(weights.values()),1.) or any(v<0 for v in weights.values()):
        raise ValueError('Invalid fixed distance weights')
    out={k:float(v) for k,v in weights.items() if k!='dino' and v>0}
    if weights.get('dino',0)>0:
        out.update({k:float(weights['dino']*v) for k,v in DINO_WEIGHTS.items()})
    return out

def distance(x,z):
    return np.maximum(np.square(x).sum(1)[:,None]+np.square(z).sum(1)[None,:]-2*x@z.T,0.)

def kernel(a,b,weights,gamma,guard=None):
    result=None
    for name,weight in expanded_weights(weights).items():
        if guard:guard.check()
        part=distance(a[name],b[name]);part*=weight
        if result is None:result=part
        else:result+=part
    result*=(-float(gamma));np.exp(result,out=result)
    return result

def fit(features,labels,weights,gamma=2.,alpha=.3,guard=None):
    y=np.asarray(labels,dtype=np.float64)
    if y.ndim!=1 or not np.isfinite(y).all() or len(y)<2:
        raise ValueError('Fit labels must be finite training labels')
    if any(len(features[k])!=len(y) for k in expanded_weights(weights)):
        raise ValueError('Training feature/label length mismatch')
    k=kernel(features,features,weights,gamma,guard)
    col=k.mean(0);grand=float(col.mean());ym=float(y.mean())
    k-=col[:,None];k-=col[None,:];k+=grand
    k=(k+k.T)*.5;k.flat[::len(k)+1]+=float(alpha)
    if guard:guard.check()
    dual=cho_solve(cho_factor(k,lower=True,check_finite=False,overwrite_a=True),y-ym,check_finite=False)
    if not np.isfinite(dual).all():raise ValueError('Nonfinite fitted coefficients')
    return {'dual':dual,'column_mean':col,'grand_mean':grand,'target_mean':ym,
            'weights':dict(weights),'gamma':float(gamma),'alpha':float(alpha),'n_fit':len(y)}

def predict(model,training,evaluation,guard=None,batch=256):
    count=len(next(iter(evaluation.values())));parts=[]
    if any(len(training[k])!=model['n_fit'] for k in expanded_weights(model['weights'])):
        raise ValueError('Saved training feature count mismatch')
    for first in range(0,count,batch):
        if guard:guard.check()
        values={k:evaluation[k][first:first+batch] for k in expanded_weights(model['weights'])}
        cross=kernel(values,training,model['weights'],model['gamma'],guard)
        cross-=cross.mean(1,keepdims=True);cross-=model['column_mean'][None,:];cross+=model['grand_mean']
        parts.append(np.maximum(cross@model['dual']+model['target_mean'],1.))
    result=np.concatenate(parts)
    if result.shape!=(count,) or not np.isfinite(result).all():raise ValueError('Nonfinite predictions')
    return result

def metrics(actual,prediction):
    error=np.asarray(prediction,dtype=np.float64)-np.asarray(actual,dtype=np.float64)
    if error.ndim!=1 or not len(error) or not np.isfinite(error).all():raise ValueError('Invalid assessment')
    return {'mse':float(np.mean(error**2)),'mae':float(np.mean(abs(error))),'bias':float(error.mean())}
