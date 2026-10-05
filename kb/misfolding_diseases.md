# Protein Misfolding Diseases Covered by the App

This file lists the diseases the app's pattern engine can flag. The engine compares simple sequence-level features (hydrophobicity, charge imbalance, beta-sheet confidence, residue composition) against published biochemical markers. A match is a screening hint, not a diagnosis.

## Alzheimer's Disease

Category: Neurodegenerative. Why the app flags it: High beta-sheet + hydrophobic core + aromatic residues (F/Y/H) — amyloid-beta fibril formation in brain. Drug target: prevent Abeta aggregation. Pattern weight in the risk score: 42.

## Parkinson's Disease

Category: Neurodegenerative. Why the app flags it: Moderate hydrophobicity + charge imbalance + beta-sheet — mirrors alpha-synuclein aggregation destroying dopamine neurons. Pattern weight in the risk score: 38.

## Lewy Body Dementia

Category: Neurodegenerative. Why the app flags it: Charge imbalance + moderate hydrophobicity + low beta-sheet — alpha-synuclein forming Lewy body inclusions in cortical neurons. Pattern weight in the risk score: 30.

## Huntington's Disease

Category: Neurodegenerative. Why the app flags it: Very high glutamine/asparagine (>15%) + low hydrophobicity + polar-rich — polyglutamine expansion drives huntingtin aggregation killing striatal neurons. Pattern weight in the risk score: 45.

## ALS (Lou Gehrig's Disease)

Category: Neurodegenerative. Why the app flags it: Glycine-rich low-complexity region + high Q/N content + low hydrophobicity — TDP-43/FUS phase separation causing motor neuron death. Pattern weight in the risk score: 40.

## Prion Disease (CJD / Fatal Familial Insomnia)

Category: Neurodegenerative. Why the app flags it: Glycine-rich + aromatic residues + charge imbalance — PrP protein signature converting from normal alpha-helix to infectious beta-sheet. Pattern weight in the risk score: 42.

## Frontotemporal Dementia (FTD)

Category: Neurodegenerative. Why the app flags it: Large charge imbalance + polar enrichment — tau protein losing structured conformation causing neurodegeneration in frontal and temporal lobes. Pattern weight in the risk score: 30.

## Multiple System Atrophy (MSA)

Category: Neurodegenerative. Why the app flags it: Charge imbalance + beta-sheet + moderate hydrophobicity — alpha-synuclein misfolding in oligodendrocytes forming glial cytoplasmic inclusions. Pattern weight in the risk score: 26.

## Transthyretin Amyloidosis (ATTR)

Category: Systemic Amyloidosis. Why the app flags it: Beta-sheet dominant + moderate hydrophobicity + balanced charge — TTR tetramer dissociation depositing amyloid in heart and peripheral nerves. Pattern weight in the risk score: 35.

## Primary Amyloidosis (AL)

Category: Systemic Amyloidosis. Why the app flags it: Beta-sheet + moderate hydrophobicity + short balanced protein — immunoglobulin light chain misfolding depositing in kidneys, heart and liver. Pattern weight in the risk score: 30.

## Dialysis-Related Amyloidosis

Category: Systemic Amyloidosis. Why the app flags it: Beta-sheet + helix + moderate hydrophobicity + balanced charge + short protein — beta-2 microglobulin accumulating in dialysis patients depositing in joints. Pattern weight in the risk score: 25.

## Type 2 Diabetes (IAPP Amyloid)

Category: Metabolic Misfolding. Why the app flags it: Short peptide + moderate hydrophobicity + high Q/N + N/S richness + polar-dominant — IAPP amyloid destroying insulin-producing pancreatic beta cells. Pattern weight in the risk score: 48.

## Cataracts (Crystallin Misfolding)

Category: Eye Disease. Why the app flags it: Beta-sheet + moderate hydrophobicity + aromatic residues + charge imbalance — crystallin protein misfolding in eye lens causing light scattering and vision loss. Pattern weight in the risk score: 28.

## Retinitis Pigmentosa

Category: Eye Disease. Why the app flags it: Very high hydrophobicity + dominant beta-sheet + long transmembrane protein — rhodopsin misfolding causing progressive photoreceptor death and blindness. Pattern weight in the risk score: 28.

## Systemic AA Amyloidosis

Category: Systemic Amyloidosis. Why the app flags it: High aromatic residues + glycine-rich + balanced charge — Serum Amyloid A misfolding. Deposits in kidneys and liver during chronic inflammation causing organ failure. Pattern weight in the risk score: 32.

## Spinocerebellar Ataxia (SCA)

Category: Neurodegenerative. Why the app flags it: Very high Q/N content (>20%) + very low hydrophobicity + polar-dominant — polyglutamine expansion in ataxin proteins causing cerebellar neuron death and progressive loss of coordination. Pattern weight in the risk score: 42.
