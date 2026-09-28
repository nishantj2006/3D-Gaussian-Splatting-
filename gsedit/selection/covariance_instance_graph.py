"""Size/orientation-aware instance growth; protection always overrides membership."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def covariances(vertices):
    scales=np.exp(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]))
    q=np.column_stack([vertices[f'rot_{i}'] for i in range(4)]).astype(float)
    norm=np.linalg.norm(q,axis=1)
    if not np.isfinite(scales).all() or np.any(norm<1e-10):
        raise ValueError('Invalid covariance')
    q/=norm[:,None]
    axes=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    return (axes*scales[:,None,:]**2)@axes.transpose(0,2,1)


def overlap_links(points,cov,anchors,candidates,*,sigma=3.,padding=0.,neighbors=16):
    """Covariance-normalized proximity, not an exact ellipsoid collision test."""
    result=np.zeros(len(points),bool)
    if not len(anchors) or not len(candidates):
        return result
    tree=cKDTree(points[anchors])
    for begin in range(0,len(candidates),2048):
        ids=candidates[begin:begin+2048]
        _,near=tree.query(points[ids],k=min(neighbors,len(anchors)),workers=-1)
        if near.ndim==1:
            near=near[:,None]
        aid=anchors[near]
        delta=points[ids,None,:]-points[aid]
        joint=cov[ids,None,:,:]+cov[aid]+np.eye(3)*(padding/sigma)**2
        joint+=np.eye(3)*1e-10
        solved=np.linalg.solve(joint,delta[...,None])[...,0]
        distance=np.sum(delta*solved,axis=-1)
        result[ids]=(distance<=sigma*sigma).any(axis=1)
    return result


def run(a):
    from utils.ply_semantic_utils import read_vertices,write_vertices
    start=time.perf_counter();out=Path(a.output_dir)
    if out.exists():
        raise FileExistsError(out)
    prior=Path(a.edit_dir);evidence=Path(a.evidence_dir)
    report=json.loads((prior/'report.json').read_text())
    header,source=read_vertices(report['source'])
    xyz=np.column_stack([source[k] for k in ('x','y','z')])
    cov=covariances(source)
    labels=np.load(prior/'instance-ids.npy')
    protected=np.zeros(len(source),bool)
    protected[np.load(prior/'protected-source-indices.npy')]=True
    identity=report['instances'][a.instance]['id']
    e=np.load(evidence/f'{a.instance}-evidence.npz')
    ratio=e['inside']/np.maximum(e['inside']+e['outside'],1e-8)
    eligible=(e['inside']>=a.min_contribution)&(ratio>=a.min_agreement)&(
        e['support']>=a.min_views)&(e['silhouette']>=2)&(
        e['contradictions']<=2)&~protected&(labels==0)
    new=np.zeros(len(source),bool);rounds=[]
    for i in range(a.rounds):
        anchor=np.flatnonzero(labels==identity)
        proposal=np.flatnonzero(eligible&(labels==0))
        linked=overlap_links(xyz,cov,anchor,proposal,sigma=a.sigma,
            padding=report['instances'][a.instance]['radius'])
        added=linked&eligible&(labels==0)&~protected
        labels[added]=identity;new|=added
        rounds.append(int(added.sum()))
        if not added.any():
            break
    selected=np.flatnonzero((labels>0)&(labels<=len(report['instances'])))
    if protected[selected].any() or len(selected)>.15*len(source):
        raise ValueError('Protection/size gate failed')
    keep=np.ones(len(source),bool);keep[selected]=False
    out.mkdir(parents=True)
    write_vertices(out/'candidate.ply',source[keep],header,
        ['UNAPPROVED covariance-aware instance removal'])
    for name,values in [('selected-indices',selected),('instance-ids',labels),
        ('protected-source-indices',np.flatnonzero(protected)),
        ('retained-source-indices',np.flatnonzero(keep)),('added-indices',np.flatnonzero(new))]:
        np.save(out/f'{name}.npy',values)
    np.savez_compressed(out/'deleted-records.npz',source_ids=selected,vertices=source[selected])
    for name,instance in report['instances'].items():
        np.save(out/f'{name}-indices.npy',np.flatnonzero(labels==instance['id']))
        instance['count']=int((labels==instance['id']).sum())
    report.update(approved=False,parent=str(prior.resolve()),removed=len(selected),
        added=int(new.sum()),rounds=rounds,parameters=vars(a),
        protected_removed=0,elapsed_seconds=time.perf_counter()-start,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        peak_gpu_allocated_mb=0.)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['added','removed','rounds','elapsed_seconds','peak_rss_mb']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('edit-dir','evidence-dir','instance','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--min-agreement',type=float,default=.85)
    p.add_argument('--min-contribution',type=float,default=.05)
    p.add_argument('--min-views',type=int,default=3)
    p.add_argument('--sigma',type=float,default=3.)
    p.add_argument('--rounds',type=int,default=3)
    run(p.parse_args())
