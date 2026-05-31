# Microplastic Image Explorer Download

This repository includes a reproducible downloader for the Moore Institute for
Plastic Pollution Research / OpenAnalysis Microplastic Image Explorer dataset.

Source app:

- `https://www.openanalysis.org/microplastic_image_explorer/`
- App bundle: `https://www.openanalysis.org/microplastic_image_explorer/app.json`
- Image host used by the app: `https://d2jrxerjcsjhs7.cloudfront.net/`

The app bundle contains `image_metadata.csv`, supporting metadata tables, app
source files, and a list of image file names. The images themselves are stored
at the CloudFront URL shown above.

## Download

From the repository root:

```bash
python scripts/download_microplastic_image_explorer.py \
  --output-dir data/microplastic_image_explorer \
  --workers 24
```

The `data/` directory is ignored by git because the image data is too large for
normal repository storage.

## Output Layout

```text
data/microplastic_image_explorer/
├── images/                         # downloaded image files
├── metadata/
│   ├── app.json                    # raw Shinylive app bundle
│   ├── image_metadata.csv          # copied from the app bundle
│   └── app_bundle/                 # decoded files from app.json
└── manifests/
    ├── dataset_summary.json        # counts, source URLs, byte totals
    ├── image_manifest.csv          # one row per image with SHA-256
    └── metadata_manifest.csv       # decoded metadata/app files with SHA-256
```

## Verification

After download, check the summary:

```bash
python -m json.tool data/microplastic_image_explorer/manifests/dataset_summary.json
```

Expected current values from the app metadata:

- `record_count`: 10,182
- `unique_file_names`: 10,182
- `downloaded_image_count`: 10,182
- `failed_image_count`: 0
- `image_bytes`: 2,534,004,672
- image size: 2.534 decimal GB, or 2.360 GiB

The exact byte count can change if the upstream app or image host is updated.
Use the generated manifests for the exact downloaded snapshot.

## Metadata Fields

`image_metadata.csv` has these columns:

- `citation`
- `color`
- `morphology`
- `polymer`
- `size`
- `type`
- `researcher`
- `file_names`

The `size` field is sparse in the current app metadata, so do not treat it as a
complete physical scale annotation for every image.

## Current Download Snapshot

The local snapshot downloaded with this script contains:

- 10,182 image files in `data/microplastic_image_explorer/images/`
- 14 decoded app/metadata files in `data/microplastic_image_explorer/metadata/app_bundle/`
- 10,182 image manifest rows plus a header in `image_manifest.csv`
- total image bytes: 2,534,004,672
- on-disk folder size reported by `du -sh`: about 2.4G

Because `data/` is ignored by git, commit the downloader, manifests if desired,
and documentation, but do not commit the downloaded image files to an ordinary
GitHub repository. For sharing the snapshot, use a data archive, GitHub Release
asset, or Git LFS only if the hosting limits fit the project.

## Re-running

The downloader skips existing files by default and recomputes their SHA-256
checksums for the manifest. To force a fresh image download:

```bash
python scripts/download_microplastic_image_explorer.py \
  --output-dir data/microplastic_image_explorer \
  --workers 24 \
  --no-skip-existing
```

For a small smoke test:

```bash
python scripts/download_microplastic_image_explorer.py \
  --output-dir data/microplastic_image_explorer_test \
  --limit 25
```
