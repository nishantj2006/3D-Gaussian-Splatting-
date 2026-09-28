"""Reuse footprint evidence to expand a protected connected instance preview.

Core-volume membership uses multiple camera silhouettes and a narrow occupied
seed band, not a rectangular-box cut. All protected source records are retained.
"""
import argparse
import json
from pathlib import Path
import time
import resource
import numpy as np
from scipy.spatial import cKDTree
from gsedit.selection.spatial_instance_removal import seeded_group,occupied_envelope


def core_membership(silhouette,visible,contradictions,distance,radius,recall):
    return (visible>=3)&(silhouette>=np.ceil(visible*recall))&(
        contradictions<=2)&(distance<=radius)


def run(a):
    from gsedit.selection.multiview_instance import project
    from utils.ply_semantic_utils import read_vertices,write_vertices
    started=time.perf_counter();out=Path(a.output_dir)
    if out.exists():
        raise FileExistsError(out)
    prior=Path(a.edit_dir)
    report=json.loads((prior/'report.json').read_text())
    ply,source=read_vertices(report['source'])
    xyz=np.column_stack([source[k] for k in ('x','y','z')])
    labels=np.load(prior/'instance-ids.npy')
    protected=np.zeros(len(source),bool)
    protected[np.load(prior/'protected-source-indices.npy')]=True
    cameras={c['img_name']:c for c in json.loads(Path(a.cameras).read_text())}
    visible=np.zeros(len(source),np.uint16)
    for view in report['training_views']:
        c=cameras[view]
        visible+=project(xyz,c,(round(270*c['height']/c['width']),270))[3]
    changes={}
    for name,instance in report['instances'].items():
        identity=instance['id'];anchor=np.flatnonzero(labels==identity)
        if not len(anchor):
            changes[name]=0
            continue
        e=np.load(prior/f'{name}-evidence.npz')
        ratio=e['inside']/np.maximum(e['inside']+e['outside'],1e-8)
        distance=cKDTree(xyz[anchor]).query(xyz,workers=-1)[0]
        envelope,_=occupied_envelope(xyz,anchor,.15)
        core=core_membership(e['silhouette'],visible,e['contradictions'],
            distance,instance['radius']*a.core_radius_multiplier,a.core_recall)
        # Low-footprint agreement is allowed only inside the consistent volume.
        core &= (e['inside']+e['outside']<.05)|(ratio>=a.core_min_agreement)
        proposals=(e['inside']>=.05)&(ratio>=a.min_agreement)&(
            e['silhouette']>=2)&(e['contradictions']<=a.max_exterior_views)
        eligible=(envelope&(proposals|core))|(labels==identity)
        group=seeded_group(xyz,eligible,anchor,protected,radius=instance['radius'])
        new=group & (labels==0)
        labels[new]=identity
        changes[name]=int(new.sum())
    count=len(report['instances'])
    removed=np.flatnonzero((labels>0)&(labels<=count))
    if protected[removed].any() or len(removed)/len(source)>.15:
        raise ValueError('Protection or scene-size gate failed')
    keep=np.ones(len(source),bool);keep[removed]=False
    out.mkdir(parents=True)
    write_vertices(out/'candidate.ply',source[keep],ply,
        ['UNAPPROVED protected spatial-volume removal'])
    np.save(out/'selected-indices.npy',removed)
    np.save(out/'retained-source-indices.npy',np.flatnonzero(keep))
    np.save(out/'protected-source-indices.npy',np.flatnonzero(protected))
    np.save(out/'instance-ids.npy',labels)
    np.savez_compressed(out/'deleted-records.npz',source_ids=removed,vertices=source[removed])
    for name,instance in report['instances'].items():
        np.save(out/f'{name}-indices.npy',np.flatnonzero(labels==instance['id']))
    for name,instance in report['instances'].items():
        instance['count']=int((labels==instance['id']).sum())
    report['peak_gpu_allocated_mb']=0.
    report.update(approved=False,parent=str(prior.resolve()),added=changes,
        removed=int(len(removed)),parameters=vars(a),
        elapsed_seconds=time.perf_counter()-started,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'added':changes,'removed':len(removed),
                     'protected_removed':0,'elapsed_seconds':report['elapsed_seconds']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('edit-dir','cameras','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--min-agreement',type=float,default=.4)
    p.add_argument('--max-exterior-views',type=int,default=8)
    p.add_argument('--core-recall',type=float,default=.85)
    p.add_argument('--core-radius-multiplier',type=float,default=1)
    p.add_argument('--core-min-agreement',type=float,default=.2)
    run(p.parse_args())
