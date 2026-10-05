# How to Use the Quantum AI Protein Folding Solver

## Workflow
Enter or pick a protein sequence, give it a name and press analyze. The backend normalizes the sequence (ambiguous IUPAC codes are mapped to standard residues), runs the Chou-Fasman AI module, builds the Hamiltonian, runs VQE, then computes a disease risk score and a comparison with a healthy reference.

## Results page
The results show the sequence view with amino acid colouring, secondary structure confidence bars, a 3D structure you can rotate, the VQE energy convergence chart, the quantum state probability distribution, the energy landscape, and the disease risk panel with bullet reasoning.

## Disease risk score
The risk score runs from 0 to 100 and is built from weighted pattern matches against sixteen misfolding diseases. Levels are Healthy (known healthy reference), Low, Moderate and High. Known sequences such as amyloid-beta (score 92) and HIV protease (score 85) are looked up directly. Everything else is scored from features, so the confidence is lower. It is a screening hint and not a diagnosis.

## Mutation analysis
The mutate feature accepts notation like E22K, meaning residue 22 changes from E to K. The app reruns the analysis on the mutated sequence and shows how structure, stability and risk change. This is useful for asking which single substitution would increase or reduce aggregation propensity.

## Comparison with a reference
For known proteins the app compares dominant structure, instability index, hydrophobicity and minimum energy with the healthy reference values. For unknown proteins it compares against an ideal profile for the predicted dominant structure.

## Saved analyses and login
Users sign in with Firebase Authentication. Each analysis is stored in Firestore under the protein_results collection tied to the user id, and can be reloaded or deleted from the Saved tab.

## AI assistant and expert summary
The expert summary and the chat assistant use a large language model. With RAG enabled, they first retrieve relevant passages from the app's knowledge base and answer using those passages together with the current analysis numbers, and they show the sources used.
