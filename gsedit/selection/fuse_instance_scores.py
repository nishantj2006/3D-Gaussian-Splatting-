"""Add learned bed evidence to an existing reversible instance selection.

The classifier is never an independent deletion rule. A new splat must agree
with rendered footprints, belong to the bed's 3D component, and avoid protected
source IDs. Existing selected IDs and PLY records are preserved exactly.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.spatial import cKDTree

from gsedit.selection.covariance_instance_graph import covariances, overlap_links


def eligible_learned_bed(probabilities, evidence, protected, labels, *,
                         probability=.9, margin=.4, agreement=.75,
                         contribution=.05, views=2, silhouettes=2,
                         contradictions=2):
    """Return only independently corroborated, previously unselected bed IDs."""
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError('Expected background/bed/frame probabilities')
    n = len(labels)
    if len(probabilities) != n or any(len(evidence[k]) != n for k in evidence.files):
        raise ValueError('Evidence/source ID alignment mismatch')
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError('Invalid probabilities')
    inside = evidence['inside']
    outside = evidence['outside']
    ratio = inside / np.maximum(inside + outside, 1e-8)
    other = np.maximum(probabilities[:, 0], probabilities[:, 2])
    available = (labels == 0) & ~protected
    return available & (probabilities[:, 1] >= probability) & (
        probabilities[:, 1] - other >= margin) & (
        inside >= contribution) & (ratio >= agreement) & (
        evidence['support'] >= views) & (evidence['silhouette'] >= silhouettes) & (
        evidence['contradictions'] <= contradictions)


def run(args):
    from utils.ply_semantic_utils import read_vertices, write_vertices

    start = time.perf_counter()
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    prior = Path(args.edit_dir)
    report = json.loads((prior / 'report.json').read_text())
    source = Path(report['source'])
    header, vertices = read_vertices(source)
    n = len(vertices)
    labels = np.load(prior / 'instance-ids.npy').copy()
    selected_before = np.load(prior / 'selected-indices.npy')
    protected_ids = np.load(prior / 'protected-source-indices.npy')
    probabilities = np.load(args.probabilities)
    evidence = np.load(Path(args.evidence_dir) / 'bed-evidence.npz')
    if len(labels) != n or selected_before.size and selected_before.max() >= n:
        raise ValueError('Prior selection/source mismatch')
    protected = np.zeros(n, bool)
    protected[protected_ids] = True
    bed_label = report['instances']['bed']['id']
    anchors = np.flatnonzero(labels == bed_label)
    if not len(anchors):
        raise ValueError('No existing bed component')
    eligible = eligible_learned_bed(
        probabilities, evidence, protected, labels,
        probability=args.probability, margin=args.margin,
        agreement=args.agreement, contribution=args.contribution,
        views=args.views, silhouettes=args.silhouettes,
        contradictions=args.contradictions)
    points = np.column_stack([vertices[k] for k in ('x', 'y', 'z')])
    candidates = np.flatnonzero(eligible)
    # First discard far-away candidates cheaply. Covariance overlap then checks
    # their actual splat extent, rather than center-only Euclidean proximity.
    near = cKDTree(points[anchors]).query(points[candidates], workers=-1)[0]
    radius = float(report['instances']['bed']['radius'])
    candidates = candidates[near <= max(radius * args.radius_multiplier, 1e-6)]
    linked = overlap_links(points, covariances(vertices), anchors, candidates,
                           sigma=args.sigma, padding=radius)
    additions = np.flatnonzero(linked & eligible)
    if len(additions) > args.max_additions:
        raise ValueError('Learned expansion exceeds addition safety cap')
    if protected[additions].any() or (labels[additions] != 0).any():
        raise AssertionError('Protection/instance collision')
    labels[additions] = bed_label
    selected = np.union1d(selected_before, additions)
    if len(selected) > args.max_scene_fraction * n:
        raise ValueError('Selection exceeds scene safety fraction')
    keep = np.ones(n, bool)
    keep[selected] = False
    out.mkdir(parents=True)
    write_vertices(out / 'candidate.ply', vertices[keep], header,
                   ['UNAPPROVED learned-score/footprint/spatial fusion'])
    for name, values in [('selected-indices', selected),
                         ('retained-source-indices', np.flatnonzero(keep)),
                         ('protected-source-indices', protected_ids),
                         ('instance-ids', labels), ('learned-additions', additions)]:
        np.save(out / (name + '.npy'), values)
    for name, instance in report['instances'].items():
        np.save(out / (name + '-indices.npy'), np.flatnonzero(labels == instance['id']))
        instance['count'] = int((labels == instance['id']).sum())
    np.savez_compressed(out / 'deleted-records.npz', source_ids=selected,
                        vertices=vertices[selected])
    report.update(approved=False, stage='learned_instance_evidence_fusion',
                  parent=str(prior.resolve()), source=str(source.resolve()),
                  candidate=str((out / 'candidate.ply').resolve()),
                  selected_before=len(selected_before), learned_eligible=int(eligible.sum()),
                  local_candidates=len(candidates), learned_additions=len(additions),
                  removed=len(selected), protected_removed=0,
                  protected_raw_bed_score=int((probabilities[protected, 1] >= args.probability).sum()),
                  parameters=vars(args),
                  elapsed_seconds=time.perf_counter() - start,
                  peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  peak_gpu_allocated_mb=0.)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('selected_before', 'learned_eligible',
        'local_candidates', 'learned_additions', 'removed', 'peak_rss_mb', 'elapsed_seconds')}))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('edit-dir', 'evidence-dir', 'probabilities', 'output-dir'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--probability', type=float, default=.9)
    p.add_argument('--margin', type=float, default=.4)
    p.add_argument('--agreement', type=float, default=.75)
    p.add_argument('--contribution', type=float, default=.05)
    p.add_argument('--views', type=int, default=2)
    p.add_argument('--silhouettes', type=int, default=2)
    p.add_argument('--contradictions', type=int, default=2)
    p.add_argument('--radius-multiplier', type=float, default=2.)
    p.add_argument('--sigma', type=float, default=3.)
    p.add_argument('--max-additions', type=int, default=5000)
    p.add_argument('--max-scene-fraction', type=float, default=.15)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
