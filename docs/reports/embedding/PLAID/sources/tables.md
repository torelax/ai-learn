## Table 1

| Dataset | Passages | Tokens | Queries | Vanilla index GiB | PLAID index GiB |
| --- | --- | --- | --- | --- | --- |
| MS MARCO v1 | 8.8M | 597.9M | 6980 | 24.6 | 21.6 |
| Wikipedia | 21.0M | 2.6B | 8757 | 105.2 | 92.0 |
| LoTTE pooled | 2.4M | 339.4M | 2931 | 14.0 | 12.3 |
| MS MARCO v2 | 138.4M | 9.4B | 3903 | 246.0 | 202.2 |

## Table 2

| k | nprobe | t_cs | ndocs |
| --- | --- | --- | --- |
| 10 | 1 | 0.5 | 256 |
| 100 | 2 | 0.45 | 1024 |
| 1000 | 4 | 0.4 | 4096 |

## Table 3

| System | MRR@10 | R@100 | R@1k | 1-CPU ms | 8-CPU ms | GPU ms |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 (PISA; k=1000) | 18.7* | - | - | 8.3* | - | - |
| SPLADEv2 (PISA; k=1000) | 36.8* | - | 97.9* | 220.3* | - | - |
| ColBERTv1 | 36.1 | 87.3 | 95.2 | - | - | 54.3 |
| Vanilla ColBERTv2 (p=2,c=2^13) | 39.7 | 90.4 | 96.6 | 3485.1 | 921.8 | 53.4 |
| Vanilla ColBERTv2 (p=4,c=2^16) | 39.7 | 91.4 | 98.3 | - | 4568.5 | 259.6 |
| PLAID ColBERTv2 (k=10) | 39.4 | - | - | 185.5 | 31.5 | 11.5 |
| PLAID ColBERTv2 (k=100) | 39.8 | 90.6 | - | 222.3 | 52.9 | 20.2 |
| PLAID ColBERTv2 (k=1000) | 39.8 | 91.3 | 97.5 | 352.3 | 101.3 | 38.4 |

## Table 4

| System | Success@5 | Success@100 | 8-CPU ms | GPU ms |
| --- | --- | --- | --- | --- |
| DPR | 66.8 | 85.0 | - | - |
| ColBERT-QA uncompressed | 75.3 | 89.2 | - | - |
| Vanilla ColBERT-QA (p=4,c=2^15; residual compression) | 74.3 | 89.0 | 5077.9 | 204.1 |
| PLAID ColBERT-QA (k=10; residual compression) | 73.3 | - | 67.1 | 13.6 |
| PLAID ColBERT-QA (k=100; residual compression) | 74.1 | 88.0 | 120.1 | 26.9 |
| PLAID ColBERT-QA (k=1000; residual compression) | 74.4 | 88.9 | 228.4 | 55.3 |

## Table 5

| System | Success@5 | Success@100 | 8-CPU ms | GPU ms |
| --- | --- | --- | --- | --- |
| BM25 | 47.8* | 77.6* | - | - |
| SPLADEv2 | 67.0* | 89.0* | - | - |
| Vanilla ColBERTv2 (p=2,c=2^13) | 69.3 | 90.3 | 1508.4 | 66.9 |
| PLAID ColBERTv2 (k=10) | 69.1 | - | 35.5 | 9.2 |
| PLAID ColBERTv2 (k=100) | 69.4 | 89.9 | 64.8 | 17.4 |
| PLAID ColBERTv2 (k=1000) | 69.6 | 90.5 | 163.1 | 27.3 |

## Table 6

| System | MRR@100 | R@100 | R@1k | 8-CPU ms | GPU ms |
| --- | --- | --- | --- | --- | --- |
| BM25 (Anserini; augmented) | 8.7 | 40.3 | 69.3 | - | - |
| Vanilla ColBERTv2 (p=4,c=2^16) | 18.0 | 68.2 | 88.1 | 5228.5 | OOM |
| PLAID ColBERTv2 (k=10) | - | - | - | 136.4 | 47.1 |
| PLAID ColBERTv2 (k=100) | 17.9 | 67.0 | - | 181.9 | 96.1 |
| PLAID ColBERTv2 (k=1000) | 18.0 | 68.4 | 85.7 | 251.3 | OOM |
