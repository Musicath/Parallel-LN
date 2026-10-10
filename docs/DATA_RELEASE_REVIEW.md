# Data-release review and author decision

This record covers the nine CSVs in [`DATA_MANIFEST.sha256`](DATA_MANIFEST.sha256)
for the planned arXiv code release. On 2026-10-10, the author confirmed that
**all nine candidate CSVs may be published under the repository's Apache-2.0
license**, including the fixed 6k index/random-label CSV. This records the
author's decision; it is not an independent legal determination of ownership
or of third-party dataset terms. No raw
CIFAR-10 images, original W&B logs, or model checkpoints are included in these
CSVs. Byte hashes in the manifest identify the candidate files, not rights to
redistribute them or proof that the historical experiments are reproducible.

| Candidate files | Rows | Contents and source | Release review |
| --- | ---: | --- | --- |
| `figures/data/dense_fourier_d4_runs.csv`, `dense_fourier_d8_runs.csv` | 720 each | Configurations and numerical outputs for author-generated synthetic Fourier targets; private paths and device strings were removed. | Author approved Apache-2.0 release; final exact-file review remains. |
| `figures/data/dense_fourier_d4_table.csv`, `dense_fourier_d8_table.csv` | 16 each | Author-computed summaries of the synthetic Fourier runs. | Author approved Apache-2.0 release; retain provenance of underlying runs. |
| `figures/data/cnn_runs.csv`, `vit_runs.csv` | 288; 1,377 | Per-run configurations and numeric CIFAR-10 fitting/classification metrics, with relative identifiers of archived run configurations; no images or per-example labels. | Author approved Apache-2.0 release; cite CIFAR-10 separately. These are not copies of the dataset. |
| `experiments/pinn/data/table2_selected_runs.csv`, `ntk_matched_240.csv` | 42; 240 | Sanitized configurations and numeric PINN/NTK results from author-side records; raw W&B metadata and absolute host paths were omitted. | Author approved Apache-2.0 release; retain the two distinct protocols in the PINN guide. |
| `experiments/vit/data/fixed_6k_labels.csv` | 6,000 | The first 6,000 CIFAR-10 training indices and the experiment's fixed random labels. The CSV no longer contains CIFAR-10's original clean class labels; those are read from the user's own dataset copy and checked against an archived digest. | Author expressly approved Apache-2.0 release of this minimized file; it does not license CIFAR-10 itself. |

The [official CIFAR-10 page](https://cave.cs.toronto.edu/kriz/cifar.html)
identifies the dataset creators and requests citation of Alex Krizhevsky's
technical report when the dataset is used. The repository does not bundle its
images or original per-example labels. The fixed-label builder obtains those
labels from the user's CIFAR-10 copy and checks their first-6k tensor digest.
This reduces third-party data in the candidate but is **not** a legal ruling
on the dataset or a grant of rights to publish the experimental records.

The previous three-column manifest is preserved only in ignored local
`outputs/fixed_6k_labels_with_clean_20261010.csv` (SHA-256
`0532e4445f4e51889dd0c9494d34f6ab6cf55469b89753c92fc63100c067c467`),
not in the candidate release. The two-column replacement has SHA-256
`b15f79b592e51f549bd4dd1d90abdb331667eed8296d9429c9d31e8eeeac65b8`.
On the eight-RTX-4090 host, both versions rebuilt 6k and 20k artifacts from
the **same** existing CIFAR-10 cache. All four `.pt` tensors, `summary.json`,
and `metadata.json` matched byte-for-byte for each size (12 files total).
The matching metadata SHA-256 digests were
`7a7bb327f0fb96cbe6cb0257988c42550d2702e127b0a94759ce79bc093df54b`
for 6k and `44af1d5e77ac0a678fe947fd81900d984ec3a3cdcfb5d7062d3f4bffac5aa109`
for 20k. This establishes equality of generated artifacts under the tested
dataset/runtime, not independent recovery of the historical 20k `.pt` file.

Before the public push:

1. Retain the author-confirmed Apache-2.0 decision and the separate CIFAR-10
   attribution in the final release documentation. Do not imply that this
   repository relicenses the CIFAR-10 dataset or raw third-party materials.
2. Recheck the exact files and outgoing Git history for private metadata,
   credentials, and unintended dataset payloads immediately before any push.

The author lifted the publication hold on 2026-10-10 conditional on final
file and source checks. The data approval does not waive those checks.
