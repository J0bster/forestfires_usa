# Texas Wildfires and Drought Data Analysis

## Overview

This repository contains datasets and scripts for analyzing wildfires and drought conditions in Texas. Below are descriptions of the datasets, their columns, and instructions for integrating additional datasets.

---

## Datasets

### 1. `texas_fires.csv`
**Source**: [188 Million US Wildfires on Kaggle](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires?resource=download)
**Description**: Contains details about wildfires in Texas.

**Columns**:
- `FOD_ID`: Global unique identifier.
- `FIRE_SIZE`: Estimate of acres within the final perimeter of the fire.
- `FIRE_SIZE_CLASS`: Code for fire size based on acres:
  - **A**: > 0 but ≤ 0.25 acres
  - **B**: 0.26 - 9.9 acres
  - **C**: 10.0 - 99.9 acres
  - **D**: 100 - 299 acres
  - **E**: 300 - 999 acres
  - **F**: 1000 - 4999 acres
  - **G**: 5000+ acres
- `LATITUDE` / `LONGITUDE`: Latitude/Longitude (NAD83) for the fire's location (decimal degrees).
- `FIRE_YEAR`: Calendar year when the fire was discovered.
- `DISCOVERY_DATE`: Date when the fire was discovered or confirmed to exist.
- `DISCOVERY_DOY`: Day of year when the fire was discovered.
- `DISCOVERY_TIME`: Time of day the fire was discovered.
- `County`: County where the fire burned or originated.
- `STAT_CAUSE_DESCR`: Description of the statistical cause of the fire.
- `CONT_DATE`: Date when the fire was declared contained (MM/DD/YYYY).
- `CONT_DOY`: Day of year the fire was contained.
- `CONT_TIME`: Time of day the fire was contained (HHMM).
- `DISCOVERY_DATE_NEW`: Date in `YYYY-MM-DD` format.

---

### 2. `drought_county_tx.csv`
**Source**: [US Drought Monitor - County Data](https://droughtmonitor.unl.edu/DmData/DataDownload/DSCI.aspx)
**Description**: Contains drought severity data at the county level for Texas.

**Columns**:
- `State`: Always "TX".
- `County`: County name.
- `FIPS`: Federal Information Processing Standards code that uniquely identifies counties in the USA.
- `MapDate`: Date of the drought data in `YYYYMMDD` format.
- `DSCI`: Drought Severity and Coverage Index, a measure of drought intensity and spatial coverage.

---

### 3. `drought_state_tx.csv`
**Source**: [US Drought Monitor - State Data](https://droughtmonitor.unl.edu/DmData/DataDownload/DSCI.aspx)
**Description**: Contains drought severity data at the state level for Texas.

**Columns**:
- `Name`: Always "Texas".
- `MapDate`: Date of the drought data in `YYYYMMDD` format.
- `DSCI`: Drought Severity and Coverage Index, a measure of drought intensity and spatial coverage.

---

### 4. `texas_fires_with_fips.csv`
**Source**: [188 Million US Wildfires on Kaggle](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires?resource=download)
**Description**: Similar to `texas_fires.csv` but includes county-level `FIPS` codes.

**Columns**:
- `FOD_ID`: Global unique identifier.
- `FIRE_SIZE`: Estimate of acres within the final perimeter of the fire.
- `FIRE_SIZE_CLASS`: Code for fire size based on acres (same as `texas_fires.csv`).
- `LATITUDE` / `LONGITUDE`: Latitude/Longitude (NAD83) for the fire's location (decimal degrees).
- `FIRE_YEAR`: Calendar year when the fire was discovered.
- `DISCOVERY_DATE`: Date when the fire was discovered or confirmed to exist.
- `DISCOVERY_DOY`: Day of year when the fire was discovered.
- `DISCOVERY_TIME`: Time of day the fire was discovered.
- `County`: County where the fire burned or originated.
- `STAT_CAUSE_DESCR`: Description of the statistical cause of the fire.
- `CONT_DATE`: Date when the fire was declared contained (MM/DD/YYYY).
- `CONT_DOY`: Day of year the fire was contained.
- `CONT_TIME`: Time of day the fire was contained (HHMM).
- `DISCOVERY_DATE_NEW`: Date in `YYYY-MM-DD` format.
- `FIPS`: Federal Information Processing Standards code that uniquely identifies counties in the USA.

## Requirements
Make sure you have the following installed to run the analysis scripts:
- Python 3.x
- Required libraries:
  ```bash
  pip install pandas numpy matplotlib
  pip install seaborn
  pip install scipy
  pip install matplotlib
  ```

Make sure you have the following installed to run the map visualization:
- Required libraries:
  ```
  pip install folium
  pip install geopandas
  pip install branca
  pip install esda
  pip install libpysal
  ```

To run the maps.py you also need a Geojson shape file of the county boundaries of Texas. You can download it from [here](https://gis-txdot.opendata.arcgis.com/datasets/TXDOT::texas-county-boundaries-detailed/explore?location=30.834886%2C-100.077018%2C6.22). The file needs to be placed in the dataset_gitignore folder with the name County_shape.geojson.
