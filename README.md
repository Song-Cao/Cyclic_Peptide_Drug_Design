
# Designing Orally Available Cyclic Peptide

This repository implements a comprehensive pipeline for designing structurally diverse cyclic peptides with oral availability, utilizing advanced machine learning techniques including diffusion, contrastive learning, and reinforcement learning. The pipeline integrates molecular and protein data for the generation and optimization of cyclic peptide candidates with enhanced binding and pharmacokinetic properties.

## Features

- **Data Preprocessing**: Handles raw molecular and protein data, cleaning, and feature extraction.
- **Contrastive Learning**: Learns joint embeddings of molecular and protein features to capture structural and functional relationships.
- **Diffusion Models**: Simulates the generation of optimized peptide candidates through a diffusion-based generative model.
- **Reinforcement Learning**: Optimizes generated candidates with a policy-gradient-based approach, guided by a reward predictor for peptide properties.
- **Integration**: Seamlessly connects all components into an end-to-end training pipeline.


---

## Usage

1. **Integrating molecular and Protein Data**:
   ```bash
   python mol_peptide_data_querying.py
   ```
   This script cleans raw molecular and protein data, and saves processed features in JSON, HDF5, and NPY formats.

2. **Construct/pretrain the contrastive model**:
   ```bash
   python contrastive_model.py
   ```

3. **Construct/pretrain the diffusion model**:
   ```bash
   python diffusion_model.py
   ```

4. **Run reinforcement framework**:
   ```bash
   python reinforcement_learning_framework.py
   ```

5. **Integrated training**:
   To run the entire pipeline:
   ```bash
   python integrated_training.py
   ```

---

## License

This project is licensed under the [MIT](LICENSE) file included in the repository. Redistribution and use require prior permission.

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request for review.


