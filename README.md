# Microplastic Image Explorer Dataset Tools

Code for downloading the OpenAnalysis / Moore Institute Microplastic Image
Explorer dataset, filtering its metadata, and visualizing the morphology of
records that include particle-size metadata.

Dataset source: <https://www.openanalysis.org/microplastic_image_explorer/>

The full image dataset is about 2.53 GB and is downloaded into `data/`, which is
ignored by git.

## Install

```bash
python -m pip install -r requirements.txt
```

## Download

Metadata only:

```bash
python scripts/download_image_explorer.py \
  --output-dir data/microplastic_image_explorer
```

Metadata plus all 10,182 images:

```bash
python scripts/download_image_explorer.py \
  --output-dir data/microplastic_image_explorer \
  --download-images \
  --workers 24
```

## Visualize Size-labeled Morphology

```bash
python scripts/visualize_metadata.py \
  --metadata data/microplastic_image_explorer/metadata/image_metadata.csv \
  --images-dir data/microplastic_image_explorer/images \
  --output-dir docs/assets
```

This creates:

- particle-size metadata availability
- morphology counts among records with particle-size metadata
- one montage for each morphology among records with particle-size metadata

## 1. Particle-size Metadata Availability

![Particle-size metadata availability](docs/assets/size_availability.png)

## 2. Morphologies Among Records With Particle-size Metadata

![Morphologies among records with particle-size metadata](docs/assets/size_labeled_morphology_counts.png)

## 3. Morphology Montages for Size-labeled Records

### Fiber

![Fiber montage](docs/assets/morphology_montage_fiber.jpg)

### Fragment

![Fragment montage](docs/assets/morphology_montage_fragment.jpg)

### Sphere

![Sphere montage](docs/assets/morphology_montage_sphere.jpg)

### Film

![Film montage](docs/assets/morphology_montage_film.jpg)

### Pellet

![Pellet montage](docs/assets/morphology_montage_pellet.jpg)

### Blank Morphology

![Blank morphology montage](docs/assets/morphology_montage_blank.jpg)

### Foam

![Foam montage](docs/assets/morphology_montage_foam.jpg)

### Other

![Other morphology montage](docs/assets/morphology_montage_other.jpg)

### Fiber Bundle

![Fiber bundle montage](docs/assets/morphology_montage_fiber_bundle.jpg)

## Filter Metadata

Example: filter fiber records with particle-size metadata:

```bash
python scripts/filter_metadata.py \
  --metadata data/microplastic_image_explorer/metadata/image_metadata.csv \
  --has-size \
  --morphology fiber \
  --output outputs/size_labeled_fibers.csv \
  --write-urls outputs/size_labeled_fiber_urls.txt
```

## Scale Note

The `size` field is particle-size metadata, not image pixel calibration. It is
sparse and mixed-format. Do not assume the dataset provides microns-per-pixel
scale for every image.

Read [docs/microplastic_image_explorer_metadata.md](docs/microplastic_image_explorer_metadata.md)
for details on particle-size metadata, magnification fields in supporting
tables, and safe wording for papers.
