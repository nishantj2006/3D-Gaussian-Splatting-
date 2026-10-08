"""Official GaussianCut solver benchmark with local footprint/PLY adapters.

Not an unmodified end-to-end GaussianCut reproduction: segmentation evidence
comes from this project's non-held-out rasterized masks, and protected source
IDs receive hard sink capacities. The upstream graph energy/solver is unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from gsedit.runtime import PROJECT_ROOT


def terminal_evidence(inside, outside, support, protected, threshold=.9):
    total = inside + outside
    ratio = np.divide(inside, total, out=np.zeros_like(inside, dtype=float), where=total > 1e-8)
    source = (ratio >= threshold) & (support >= 3) & (inside >= .05) & ~protected
    sink = ((ratio < .1) & (total >= .05)) | protected
    if source.sum() < 5 or sink.sum() < 5:
        raise ValueError('Insufficient independent positive/negative evidence')
    ws = ratio.astype(np.float32)
    wt = (1-ratio).astype(np.float32)
    ws[protected] = 0
    # Strong negative capacities; post-solver source-ID protection is also checked.
    wt[protected] = 1e6
    return ws, wt, source, sink


class SolverGaussians:
    """CPU-only adapter; save full schema ourselves from the returned source IDs."""
    def __init__(self, vertices):
        import torch
        self._xyz = torch.from_numpy(np.column_stack([vertices[k] for k in ('x','y','z')]))
        self._features_dc = torch.from_numpy(np.column_stack([vertices[f'f_dc_{i}'] for i in range(3)])[:,None,:])
        for attr, prefix, count in [('_features_rest','f_rest_',45), ('_scaling','scale_',3), ('_rotation','rot_',4)]:
            setattr(self, attr, torch.from_numpy(np.column_stack([vertices[f'{prefix}{i}'] for i in range(count)])))
        self._opacity = torch.from_numpy(vertices['opacity'].copy()[:,None])

    def save_ply(self, path):
        # Upstream calls this on its geometry-only facade. Its PLY writer drops
        # semantic properties; deliberately defer to our source-record writer.
        pass


def run(a):
    import torch
    from utils.ply_semantic_utils import read_vertices, write_vertices
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    prior = Path(a.edit_dir)
    previous = json.loads((prior/'report.json').read_text())
    header, vertices = read_vertices(previous['source'])
    protected_ids = np.load(prior/'protected-source-indices.npy')
    protected = np.zeros(len(vertices), bool)
    protected[protected_ids] = True
    evidence_path = Path(a.evidence_dir)/'bed-evidence.npz'
    evidence = np.load(evidence_path)
    ws, wt, positive, negative = terminal_evidence(
        evidence['inside'], evidence['outside'], evidence['support'], protected)
    if a.seed_current:
        seeds = np.load(prior/'bed-indices.npy')
        seeds = seeds[~protected[seeds]]
        positive[seeds] = True
        negative[seeds] = False
        ws[seeds] = 1.
        wt[seeds] = 0.
    repo = Path(a.upstream_dir).resolve()
    path = repo/'gaussian-splatting/utils/graphcut.py'
    spec = importlib.util.spec_from_file_location('upstream_gaussiancut', path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    out.mkdir(parents=True)
    (out/'graphcut_benchmark').mkdir()
    np.save(out/'positive-source-indices.npy', np.flatnonzero(positive))
    np.save(out/'negative-source-indices.npy', np.flatnonzero(negative))
    graph_params = SimpleNamespace(num_edges=10, terminal_clusters_source=5,
                                   terminal_clusters_sink=5, leaf_size=40)
    args = SimpleNamespace(identifier='benchmark', sig_pos_neigh=1., sig_col_neigh=1.,
        sig_pos_term=.1, sig_col_term=1., weight_pos=.5, weight_color=.5,
        user_weight_term=a.user_weight, cluster_term=1.)
    print(json.dumps({'stage':'official_solver','scene_splats':len(vertices),
        'positive_seeds':int(positive.sum()),'negative_seeds':int(negative.sum()),
        'protected':len(protected_ids),'upstream_commit':commit}), flush=True)
    graph_start = time.perf_counter()
    _, background = upstream.graphcut_segmentation(args, str(out), graph_params,
        torch.from_numpy(ws[:,None]), torch.from_numpy(wt[:,None]),
        SolverGaussians(vertices), SolverGaussians(vertices[positive]), SolverGaussians(vertices[negative]))
    graph_seconds = time.perf_counter()-graph_start
    bed_ids = np.flatnonzero(background == 0)  # segment 0 is upstream source/foreground
    if protected[bed_ids].any():
        raise AssertionError('Hard protected sink was selected by solver')
    frame_ids = np.load(prior/'wooden_frame-indices.npy')
    selected = np.union1d(bed_ids, frame_ids)
    labels = np.zeros(len(vertices), np.int32)
    labels[bed_ids] = 1
    labels[np.setdiff1d(frame_ids, bed_ids)] = 2
    safe = len(selected) <= .15*len(vertices)
    kept = np.setdiff1d(np.arange(len(vertices)), selected)
    candidate = out/('candidate.ply' if safe else 'rejected-diagnostic.ply')
    write_vertices(candidate, vertices[kept], header, ['UNAPPROVED official GaussianCut solver, local evidence adapter'])
    for name, data in [('selected-indices',selected), ('bed-indices',bed_ids),
        ('wooden_frame-indices',np.flatnonzero(labels==2)), ('instance-ids',labels),
        ('retained-source-indices',kept), ('protected-source-indices',protected_ids)]:
        np.save(out/f'{name}.npy', data)
    np.savez_compressed(out/'deleted-records.npz', source_ids=selected, vertices=vertices[selected])
    report = dict(previous)
    report.update(source=previous['source'], candidate=str(candidate), approved=False,
        method='Official GaussianCut graphcut_segmentation with cached local footprint evidence and hard protected sinks',
        upstream_commit=commit, upstream_code_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parameters=vars(a), graph_parameters=vars(graph_params), energy_parameters=vars(args),
        protected_removed=0, removed=len(selected), bed_removed=len(bed_ids),
        positive_seeds=int(positive.sum()), negative_seeds=int(negative.sum()),
        size_gate_passed=safe, source_properties=len(vertices.dtype.names),
        semantic_dimensions=sum(k.startswith('semantic_') for k in vertices.dtype.names),
        retained_records_exact=np.array_equal(read_vertices(candidate)[1],vertices[kept]),
        elapsed_seconds=time.perf_counter()-started, graph_seconds=graph_seconds,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        peak_gpu_allocated_mb=0., evidence_cache=str(evidence_path.resolve()),
        caveats=['Consult the evidence stage report for intact versus previously pruned opacity; heldouts are excluded.',
                 'Hard protection is a safety adaptation, not an upstream default.',
                 'Automatic masks are pseudo-labels and protected masks may cover multiple furniture instances.',
                 'No reconstruction or image generation in this stage.'])
    for name, detail in report['instances'].items():
        detail['count'] = int((labels==detail['id']).sum())
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['removed','bed_removed','size_gate_passed','graph_seconds','elapsed_seconds','peak_rss_mb']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('edit-dir','evidence-dir','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--upstream-dir',default=str(PROJECT_ROOT/'external_tools/GaussianCut'))
    p.add_argument('--user-weight',type=float,default=1.)
    p.add_argument('--seed-current',action='store_true')
    run(p.parse_args())
