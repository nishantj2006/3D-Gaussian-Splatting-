"""Reason tagging is explicit about overlapping, non-exclusive causes."""
from gsedit.evaluation.diagnose_current_bed_residue import reason_tags


def test_protected_broad_splat_reports_all_relevant_reasons():
    tags=reason_tags(True,False,3.,.35,.08,False,3)
    assert tags==['protected_source','outside_local_bed_group',
                  'mixed_bed_background_footprint','weak_learned_bed_score']


def test_local_high_evidence_is_not_silently_called_bed():
    assert reason_tags(False,False,.2,.95,.9,False,3)==['unresolved_high_evidence']
