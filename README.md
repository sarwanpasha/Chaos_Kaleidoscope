# Chaos Kaleidoscope (DANCE)


A Python tool for generating beautiful kaleidoscope-like visualizations of protein sequences using Chaos Game Representation (CGR). This project transforms amino acid sequences into symmetric geometric patterns that can reveal structural patterns and be used for machine learning applications.

## Features

- **Chaos Game Representation (CGR)**: Maps each of the 20 standard amino acids to specific coordinates in 2D space
- **Kaleidoscopic Visualization**: Creates symmetric, fractal-like patterns through recursive mirroring
- **Batch Processing**: Automatically processes multiple protein sequences and organizes outputs by classification labels
- **Customizable Parameters**: Adjustable recursion depth, scale, colors, and positioning

## Prerequisites

```bash
pip install numpy matplotlib scikit-learn
```

## Required Data Files

The script expects two NumPy arrays:
- `sequence_data.npy`: Array containing protein sequences as strings
- `attributes_data.npy`: Array containing classification labels for each sequence

## How It Works

### Chaos Game Representation (CGR)

Each amino acid is mapped to specific coordinates:

| Amino Acid | Coordinates | Amino Acid | Coordinates |
|------------|-------------|------------|-------------|
| A (Alanine) | (0.5, 0.5) | L (Leucine) | (0.75, 0.0) |
| C (Cysteine) | (1.0, 0.5) | M (Methionine) | (0.5, 0.0) |
| D (Aspartic acid) | (0.5, 1.0) | N (Asparagine) | (0.25, 0.5) |
| E (Glutamic acid) | (0.0, 0.5) | P (Proline) | (1.0, 0.0) |
| F (Phenylalanine) | (1.0, 1.0) | Q (Glutamine) | (0.0, 1.0) |
| G (Glycine) | (0.25, 0.25) | R (Arginine) | (0.5, 0.25) |
| H (Histidine) | (0.75, 0.25) | S (Serine) | (0.75, 0.5) |
| I (Isoleucine) | (0.75, 0.75) | T (Threonine) | (0.5, 0.75) |
| K (Lysine) | (0.25, 0.75) | V (Valine) | (0.0, 0.0) |
| W (Tryptophan) | (1.0, 0.25) | Y (Tyrosine) | (1.0, 0.75) |

### Kaleidoscope Generation

The algorithm creates symmetric patterns by:
1. Processing each amino acid in sequence
2. Drawing lines from current position to the amino acid's CGR coordinates
3. Creating 4-fold symmetry by mirroring across both axes
4. Recursively applying the pattern at multiple scales and positions

## Usage

### Basic Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Load your data
seq_data = np.load("sequence_data.npy")
attribute_data = np.load("attributes_data.npy")

# Generate visualization for a single sequence
protein_sequence = 'ACQRSTAGTACGT'
depth = 4
scale = 10

plt.figure(figsize=(8, 8))
generate_kaleidoscope(protein_sequence, depth, (0, 0), 0, scale, 'blue')
plt.axis('off')
plt.show()
```

### Batch Processing

The script automatically processes all sequences in your dataset:

```python
# This will create directories named after your labels
# and save individual PNG files for each sequence
images = []
labels = []

for i in range(len(seq_data)):
    sequence = seq_data[i]
    label = attribute_data[i]
    
    plt.figure(figsize=(8, 8))
    generate_kaleidoscope(sequence, depth, (0, 0), 0, scale, 'blue')
    plt.axis('off')
    
    # Saves to {label}/{index}.png
    image_path = f'{label}/{len(images)}.png'
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path)
    plt.close()
    
    images.append(image_path)
    labels.append(label)
```

## Parameters

- **depth**: Recursion depth (higher values create more complex patterns)
- **scale**: Size scaling factor for the visualization
- **position**: Starting coordinates (typically (0, 0))
- **angle**: Initial rotation angle
- **color**: Line color for the visualization

## Output

The script generates:
- Individual PNG images organized in directories by classification label
- Symmetric, kaleidoscope-like patterns unique to each protein sequence
- Images suitable for machine learning applications or visual analysis

## Applications

- **Protein Classification**: Visual patterns can be used as features for ML models
- **Sequence Analysis**: Identify structural similarities between proteins
- **Data Visualization**: Create artistic representations of biological data
- **Pattern Recognition**: Detect recurring motifs in protein sequences

## File Structure

```
project/
├── sequence_data.npy          # Input protein sequences
├── attributes_data.npy        # Classification labels
├── protein_visualization.py   # Main script
└── output/
    ├── class1/
    │   ├── 0.png
    │   ├── 1.png
    │   └── ...
    └── class2/
        ├── 0.png
        ├── 1.png
        └── ...
```

## Examples

### Simple Sequence
A sequence of all alanines (`AAAAAAAAAAAAA`) creates a regular, symmetric pattern.

### Complex Sequence
A mixed sequence (`ACQRSTAGTACGT`) generates a more intricate, unique kaleidoscope pattern.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the [MIT License](LICENSE).

## Citation

If you use this visualization method in your research, please cite:

```
@article{murad2024dance,
  title={Dance: Deep learning-assisted analysis of protein sequences using chaos enhanced kaleidoscopic images},
  author={Murad, Taslim and Chourasia, Prakash and Ali, Sarwan and Khan, Imdad Ullah and Patterson, Murray},
  journal={arXiv preprint arXiv:2409.06694},
  year={2024}
}
```
