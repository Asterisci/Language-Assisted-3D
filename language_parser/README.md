# Language Parser

The language parser is based on an off-the-shelf nlp toolkit [language scene graph parse](https://github.com/vacancy/SceneGraphParser). We add additional rules to the original toolkit (see `sng_parser/backends/spacy_parser.py`).

## Installation(optional)

**Corresponding packages should be installed** if following our [Setup](../README.md#setup) Section, or you can install them separately. (We test the code on Python3.8)

```python
pip install tabulate spacy
python -m spacy download en_core_web_sm
```

## Data preparation
ScanRefer dataset should be downloaded if you should follow our [Dataset](../README.md#dataset) Section. 
Put `ScanRefer_filtered_train.json` and `ScanRefer_filtered_val.json` under the `data` folder.

## Usage

After preparation, extract attribution and relation information from ScanRefer Dataset by running:

```python
python parser.py
```
There should be the following files:

- `ScanRefer_filtered_{SPLIT}_parser.json` : combined ScanRefer dataset with extracted information.
- (For debug)`ScanRefer_filtered_{SPLIT}_rel_failinfo.json` : failed extract samples.