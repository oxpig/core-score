"""Split ensembles with several identical ligand copies into one ensemble per site."""

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from core_score.analysis import ost
from core_score.outputs.files import diagnostic, pose_directory, write_json
from core_score.inputs.structures import filter_records, read_document


def prepare_site_groups(groups, output, *, timeout=None):
    """One ensemble per ligand site. Ensembles with one copy pass through as site 1.

    Each site ensemble is named for the output: the ensemble's own name when it has one
    ligand copy, and <ensemble>_site_<n> for each site when it has several."""
    split, reports = [], []
    for group in groups:
        try:
            failed = [pose["pose_id"] for pose in group["poses"] if pose.get("error")]
            if failed:
                raise ValueError(f"Poses failed preparation: {failed}; see their notes in predictions.csv")
            counts = {pose["prepared"]["copies"] for pose in group["poses"]}
            if len(counts) != 1:
                # Copies can only be matched to sites when every pose has the same number.
                raise ValueError(f"Poses have different numbers of ligand copies: {sorted(counts)}")
            copies = counts.pop()
            if copies == 1:
                split.append(dict(group, site="1", name=group["ensemble"]))
                continue
            sites, report = split_sites(group, copies, output, timeout)
            split += sites
            reports.append(report)
        except Exception as exc:
            report = diagnostic("site_assignment", exc, ensemble=group["ensemble"])
            reports.append(report)
            split.append(dict(group, site="unresolved", name=group["ensemble"], error=report))
    if reports:
        write_json(output / "sites.json", reports)
    return split


def split_sites(group, copies, output, timeout=None):
    """One ensemble per ligand site, and a report of how each copy was assigned.

    Every pose is superposed onto the first (anchor) pose by its protein, then each
    pose's copy centres are matched one-to-one to shared site centres. Site labels are
    arbitrary and do not prove biological correspondence.
    """
    poses = group["poses"]
    points, transforms = align_copy_centres(poses, timeout)
    sites, costs, centres = assign_sites(points, copies)
    split = []
    for site in range(copies):
        site_poses = []
        for pose in poses:
            copy = next(c for c in pose["prepared"]["copy_records"] if sites[pose["pose_id"]][c["copy_id"]] == site)
            site_poses.append(site_pose(pose, copy, pose_directory(output, pose) / f"site{site + 1}.cif"))
        split.append(dict(group, poses=site_poses, site=str(site + 1), name=f"{group['ensemble']}_site_{site + 1}"))
    report = dict(ensemble=group["ensemble"], anchor_pose=poses[0]["pose_id"], transforms=transforms,
                  centres=centres, assignments=sites, costs=costs)
    return split, report


def align_copy_centres(poses, timeout=None):
    """Each pose's ligand-copy centres, moved into the anchor pose's frame by OST superposition."""
    anchor = poses[0]["prepared"]["path"]
    points, transforms = {}, {}
    for pose in poses:
        copies = pose["prepared"]["copy_records"]
        transform = np.eye(4)
        if pose is not poses[0]:
            transforms[pose["pose_id"]] = fit = ost.superpose(pose["prepared"]["path"], anchor, timeout)
            transform = fit["transform"]
        centres = np.array([copy["centre"] for copy in copies], dtype=float)
        moved = (np.hstack([centres, np.ones((len(centres), 1))]) @ transform.T)[:, :3]
        points[pose["pose_id"]] = {copy["copy_id"]: xyz for copy, xyz in zip(copies, moved)}
    return points, transforms


def assign_sites(points, k, seed=0, max_iter=50):
    """Label each pose's k copies with k sites, exactly one copy per site per pose.

    k-means seeds the site centres. Each pose's copies are then matched one-to-one to the
    centres (Hungarian algorithm on squared distance) and the centres re-estimated, until
    the labels stop changing. Returns {pose: {copy: site}}, {pose: {copy: squared
    distance}} and the final centres.
    """
    all_points = np.vstack([np.asarray(list(copies.values())) for copies in points.values()])
    centres, _ = kmeans2(all_points, k, seed=seed, minit="++")
    labels, costs = {}, {}
    for _ in range(max_iter):
        previous = dict(labels)
        for pose_id, copies in points.items():
            xyz = np.asarray(list(copies.values()))
            cost = ((xyz[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            rows, sites = linear_sum_assignment(cost)
            labels[pose_id] = dict(zip(copies, (int(s) for s in sites)))
            costs[pose_id] = dict(zip(copies, (float(c) for c in cost[rows, sites])))
        if labels == previous:
            return labels, costs, centres
        for site in range(k):
            members = [points[p][c] for p, assigned in labels.items() for c, s in assigned.items() if s == site]
            if members:
                centres[site] = np.mean(members, axis=0)
    raise ValueError("Site assignment did not converge; no site labels accepted")


def site_pose(pose, copy, destination):
    """The pose reduced to one ligand copy, with its original coordinates."""
    prepared = dict(pose["prepared"], copies=1, copy_records=[copy])
    if copy.get("residue"):
        filter_records(read_document(prepared["path"]), {tuple(copy["residue"])}, destination)
        prepared["path"] = str(destination)
    if prepared["ligand_sdfs"]:
        # Preserve the selected chemistry when splitting multiple copies into sites.
        index = pose["prepared"]["copy_records"].index(copy)
        prepared["ligand_sdfs"] = [pose["prepared"]["ligand_sdfs"][index]]
    return dict(pose, prepared=prepared)
