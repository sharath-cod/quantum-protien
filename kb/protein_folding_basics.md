# Protein Folding Basics

## What is protein folding
A protein is a chain of amino acids (a sequence written with 20 one-letter codes such as A, G, L, K). The chain folds into a specific 3D shape, and that shape determines what the protein does. Folding is driven mostly by the hydrophobic effect: oily (hydrophobic) residues bury themselves in the core while polar and charged residues face the water.

## Levinthal's paradox
A chain of 100 residues has an astronomically large number of possible conformations. If a protein tried each one at random it would take longer than the age of the universe, yet real proteins fold in milliseconds to seconds. This means folding follows guided pathways down an energy funnel rather than a random search. It is the main reason folding prediction is hard and the motivation for trying quantum optimization.

## Four levels of structure
Primary structure is the amino acid sequence. Secondary structure is local shape: alpha helix, beta sheet, turn, random coil. Tertiary structure is the full 3D fold of one chain. Quaternary structure is how several chains assemble into a complex.

## Misfolding and aggregation
If a protein folds wrongly it can lose function or clump into aggregates. Beta-sheet-rich misfolded proteins can stack into amyloid fibrils, which are linked to Alzheimer's disease, Parkinson's disease, type 2 diabetes and systemic amyloidosis. Cells use chaperone proteins and degradation systems to fight misfolding, and these weaken with age.

## Why sequence alone is not enough
Real folding depends on the cellular environment (pH, temperature, chaperones, salt), post-translational modifications and neighbouring chains. A sequence-only tool like this app can estimate tendencies, but it cannot replace experimental structures (X-ray crystallography, cryo-EM, NMR) or deep-learning predictors such as AlphaFold.
