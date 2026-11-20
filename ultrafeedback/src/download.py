"""Pinned downloads and source records. Run in Colab after accepting source access terms."""
from datetime import datetime, timezone
from pathlib import Path
from .core import save_json


def download_dataset(spec, root):
    from huggingface_hub import HfApi, hf_hub_download
    from datasets import load_dataset
    repo, revision = spec['repo'], spec['revision']
    if not revision or revision in {'main','master'}:
        raise ValueError('Pin a commit SHA before downloading.')
    root = Path(root)
    raw = root / 'data' / 'raw' / spec['name']
    info = HfApi().dataset_info(repo, revision=revision)
    card_data = info.card_data.to_dict() if info.card_data is not None else {}
    license_value = card_data.get('license', 'unknown')
    card = hf_hub_download(repo_id=repo, filename='README.md', repo_type='dataset', revision=revision)
    raw.mkdir(parents=True, exist_ok=True)
    (raw / 'README.source.md').write_text(Path(card).read_text())
    kwargs = {'data_files':spec['data_files']} if spec.get('data_files') else {}
    ds = load_dataset(repo, revision=revision, split=spec['split'], **kwargs)
    target = raw / 'dataset'
    if not target.exists():
        ds.save_to_disk(str(target))
    save_json(raw / 'provenance.json', {'repo':repo, 'requested_revision':revision, 'resolved_revision':info.sha,
                                      'split':spec['split'], 'data_files':spec.get('data_files'),
                                      'downloaded_utc':datetime.now(timezone.utc).isoformat(), 'license_metadata':license_value,
                                      'license_review':'Review saved card and upstream source terms; metadata is not a complete license determination.',
                                      'rows':len(ds), 'fingerprint':ds._fingerprint})
    return ds
