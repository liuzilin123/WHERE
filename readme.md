# WHERE


WHERE : Weekly Hypergraph Evolving Representation Learning for Next POI Recommendation

---

## Datasets

The experiments can be conducted on location-based check-in datasets, such as:

- Foursquare (**NYC**, **TKY**)

Raw data should be placed under : 
- data/raw_data/

PreProcessing data:
- data/nyc/
- data/tky/

---

## Requirements

- Python 3.8+
- PyTorch 1.10+
- numpy
- pandas
- scipy
- scikit-learn
- tqdm

---

## Project Structure

```text
.
├── create_hypergraph.py    # build weekly multi-type hypergraphs
├── model.py                # model definition
├── main.py                 # training and evaluation
└── data/
    ├── raw_data/           # raw datasets
    ├── nyc/
    │   ├── sessions/       # train / val / test sessions
    │   ├── snapshots/      # weekly snapshots
    │   └── graph/          # constructed hypergraphs
    └── tky/
        ├── sessions/       # train / val / test sessions
        ├── snapshots/      # weekly snapshots
        └── graph/          # constructed hypergraphs

```

## Usage

### Train and evaluate the model:

```text
python main.py
```
