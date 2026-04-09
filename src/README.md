# Thesis source code

This directory contains all scripts and modules to reproduce the thesis results.
Any commands in this readme should be run from the /src/ folder unless stated otherwise.

## Installation

install sqlite and spatialite, there's many ways to do this but it is recommended to use either:

- conda (with ´--channel conda-forge´)
- OSGeo4W (GUI)

install [astral uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv init
uv sync
```

For notebooks, select the .venv directory as kernel

## Usage

### Preparation

#### DHM

Download the Flemish "Digitale Hoogtemodel" (DHM) from the following webpages:

- [DHM1](https://download.vlaanderen.be/product/59/configureer)
  - Select tiles 13, 14, 21 and 22 under "GeoTIFF"
  - Then press "Download"
- [DHM2](https://download.vlaanderen.be/product/936/configureer)
  - Select tiles 13, 14, 21 and 22 on the map, make sure it's the DSM, not DTM.
  - Then press "Download"
  
save the downloaded zips to their respective folders in "/src/downloads/DHM\<1|2\>_zips/\<downloaded zip\>"

#### Urban Atlas

Download the following datasets from [Copernicus/LMS](https://land.copernicus.eu/en/products/urban-atlas?tab=land_coverland_use)

- Urban Atlas Land Cover/Land Use 2006 (vector), Europe, 6-yearly - Pre-packaged
- Urban Atlas Land Cover/Land Use 2012 (vector), Europe, 6-yearly - Pre-packaged
- Urban Atlas Land Cover/Land Use 2018 (vector), Europe, 3-yearly - Pre-packaged
- Urban Atlas Land Cover/Land Use 2021 (vector), Europe, 3-yearly - Pre-packaged

Make sure to only download the Gent areas each time.

#### ODA File Converter

You must also download and install the ODA File Converter, responsible for converting the 3D Ghent model from the proprietary DWG format to a processable format.

#### Scripts

Scripts should be run in the following order:

1. `gathering/gather_all.py`
   - This script is responsible for gathering datasets and unzipping them
   - It may be possible this fails due to stale links, in that case you may have to download files or adjust the scripts manually. If neither solution is feasible, contact me for assistance.
2. wip
