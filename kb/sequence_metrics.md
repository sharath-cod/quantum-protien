# Sequence Metrics Explained

## Instability index
The instability index (Guruprasad et al., 1990) sums dipeptide instability weights. A value above 40 predicts an unstable protein in a test tube, below 40 predicts a stable one. It estimates in vitro stability only, so a protein above 40 can still be perfectly stable in a cell, and it says nothing about folding speed.

## Hydrophobicity and the Eisenberg scale
Hydrophobic ratio is the percentage of oily residues such as A, V, I, L, M, F, W. The app uses the Eisenberg consensus scale to score each residue. A high ratio means a strong drive to bury residues in a core, which helps folding of globular proteins but also raises the risk of aggregation when exposed, especially in membrane or fibril-forming segments.

## Isoelectric point (pI)
The pI is the pH at which the protein has no net charge. Below the pI the protein is net positive, above it net negative. Proteins are least soluble near their pI, which is why pI matters for purification and for aggregation. Acidic proteins such as insulin A-chain have low pI, basic proteins such as HIV protease have high pI.

## Charge balance
The app counts positively charged (K, R, H) and negatively charged (D, E) residues. A large imbalance can drive electrostatic repulsion or attraction between chains, and the disease engine uses charge imbalance as a marker for alpha-synuclein-like proteins.

## Molecular weight
Molecular weight is the sum of residue masses in daltons (Da), roughly 110 Da per residue on average. It helps choose lab methods, for example mass spectrometry or gel electrophoresis.

## Hydrophobic moment
Hydrophobic moment measures how strongly hydrophobic residues line up on one face of a helix. A high moment suggests an amphipathic helix that may sit at a membrane surface or bind another protein.
