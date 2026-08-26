# Dataset provenance

| File | Source | Retrieved | Rows | Notes |
|---|---|---|---|---|
| `A_Z_medicines_dataset_of_India.csv` | A–Z Medicines Dataset of India (Kaggle) | pre-existing | 253,973 | Brand, composition, MRP, pack, manufacturer |
| `generic.csv` | [janaushadhi.gov.in](https://janaushadhi.gov.in) Product Portfolio export | **2026-08-26** | 2,439 | PMBJP product list with current MRP |
| `master_medicines_final.csv` | Derived brand↔NPPA join (prior work) | pre-existing | 8,233 | 17.2% ceiling coverage; **0% notification coverage** |

## generic.csv was replaced on 2026-08-26

The previous copy had MRPs rounded to whole rupees. Against the fresh export,
**77% of the 2,439 rows differed**, with a mean absolute price error of 5.7%
and a maximum of 39%.

Some of that is rounding (`47.75` → `44.77`), and some is genuine price
movement (`Linezolid Infusion 600mg per 300ml`: 121 → 113.44).

Either way it fed straight into the savings arithmetic, which is the number a
patient would act on. The superseded file is kept as
`generic.csv.superseded-2026-08` so the difference stays auditable.

## Not yet ingested

- `nlem2022.pdf` — National List of Essential Medicines 2022, 135 pages.
  Worth parsing: NLEM membership is what determines whether a formulation falls
  under DPCO price control at all, so it would let the price engine distinguish
  "no ceiling on record" from "not price-controlled by design" — currently
  reported as the same thing.
