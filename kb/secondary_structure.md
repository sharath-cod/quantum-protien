# Secondary Structure Prediction

## Alpha helix, beta sheet, turn and coil
An alpha helix is a spiral held by hydrogen bonds between residues four positions apart. A beta sheet is formed by extended strands that pair side by side, and beta sheets are the core of amyloid fibrils. A beta turn is a short reversal that links elements. Random coil is flexible, unstructured chain.

## Chou-Fasman method
The app uses Chou-Fasman propensities. Each amino acid has an empirical tendency to appear in helices, sheets or turns, derived from known structures. The method scores a sliding window along the sequence and assigns the structure with the strongest propensity. It is a classic, fast, interpretable method, with typical accuracy around 50 to 60 percent, well below modern deep learning.

## Residue preferences used by the app
Helix formers: A, E, L, M. Sheet formers: C, F, I, V, W, Y. Turn formers: D, G, N, S. Coil formers: P, H, K, R, T. Proline breaks helices because its ring blocks the backbone geometry, and glycine is very flexible, so both tend to appear in turns and loops.

## Reading the confidence scores
The app reports percentages for Alpha Helix, Beta Sheet, Beta Turn and Random Coil. They are relative propensity scores, not probabilities from a trained model. The dominant structure is the highest score. A high beta-sheet score combined with high hydrophobicity is the signature the disease engine associates with aggregation-prone sequences.

## Cross-validation and regions
The app cross-validates helix calls using hydrophobic moment, which measures whether hydrophobic residues repeat every 100 degrees around a helix (the amphipathic pattern). It then merges neighbouring positions into helix and sheet regions so the sequence view shows contiguous segments.
