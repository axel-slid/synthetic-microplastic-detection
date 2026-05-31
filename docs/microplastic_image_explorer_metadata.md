# Metadata and Scale Notes

The Microplastic Image Explorer is useful, but the word "scale" can mean
different things. Treat these separately.

## 1. Dataset Scale

The current app metadata contains 10,182 unique image records. A full image
download is about 2.534 GB / 2.360 GiB.

Use:

```bash
python scripts/download_image_explorer.py --output-dir data/microplastic_image_explorer
python -m json.tool data/microplastic_image_explorer/manifests/dataset_summary.json
```

## 2. Particle Size Metadata

The primary table `image_metadata.csv` has a `size` column. This is particle
size metadata, not an image calibration value.

Important limitations:

- many records have a blank `size` field
- values are mixed-format strings, such as `>1 MM`, `50-250 UM`, `100-500UM`,
  `~1MM`, or numeric-looking values
- units and conventions are not fully normalized in the primary table
- the field should be treated as source metadata, not a measured pixel scale

To filter records with nonblank size metadata:

```bash
python scripts/filter_metadata.py \
  --metadata data/microplastic_image_explorer/metadata/image_metadata.csv \
  --has-size \
  --output outputs/records_with_size.csv
```

To filter for larger-size text bins:

```bash
python scripts/filter_metadata.py \
  --metadata data/microplastic_image_explorer/metadata/image_metadata.csv \
  --size-regex '>500|>1\\s*MM|2\\s*MM' \
  --output outputs/larger_size_bins.csv \
  --write-urls outputs/larger_size_bins_urls.txt
```

## 3. Pixel Scale / Microns Per Pixel

The primary metadata does not provide a reliable per-image pixel-to-micron
calibration, scale bar, or microns-per-pixel field. That means you should not
convert segmentation mask area in pixels to physical area in microns from the
primary Image Explorer metadata alone.

Do not write claims like:

> Each image has known microns-per-pixel scale.

Safer wording:

> The Image Explorer metadata includes particle-size categories for a subset of
> records, but it does not provide complete per-image pixel calibration.

## 4. Magnification and Supporting Tables

The decoded app bundle includes supporting files under:

```text
data/microplastic_image_explorer/metadata/app_bundle/extra_data/
```

Examples include:

- `tbl_microscopysettings.csv`, with fields such as `labid`, `sampleid`,
  `sizefraction`, and `magnification`
- `tbl_qa_master.csv`, with particle identifiers and quality-assurance fields
- source spreadsheets such as `Photos_data.xlsx`,
  `MethodEvaluationStudy_ALGALITA.xlsx`, and
  `Fadare and Conkle MP Taxonomy.xlsx`

These files can help interpret subsets of the data. They are not, by
themselves, a universal per-image pixel-scale table. Joining them to
`image_metadata.csv` requires source-specific logic and should be documented
for the subset being analyzed.

## Recommended Metadata Workflow

1. Download metadata first, without images.
2. Inspect `dataset_summary.json` and `image_metadata.csv`.
3. Filter records by citation, morphology, polymer, color, size text, or
   filename pattern.
4. Download only the subset needed for an experiment, or download the full
   image set with `--download-images`.
5. Keep filtered CSVs and download manifests with your analysis so every image
   can be traced back to its original URL and metadata row.

## Citation-Aware Filtering

The `citation` field identifies the data source. For example:

```bash
python scripts/filter_metadata.py \
  --metadata data/microplastic_image_explorer/metadata/image_metadata.csv \
  --citation 'Fadare and Conkle' \
  --output outputs/fadare_conkle.csv \
  --write-urls outputs/fadare_conkle_urls.txt
```

Use citation-specific subsets when you need more defensible metadata handling,
because different contributors can use different size conventions and source
tables.
