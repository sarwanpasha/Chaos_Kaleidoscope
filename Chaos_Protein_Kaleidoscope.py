import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split



seq_data = np.load("sequence_data.npy")
attribute_data = np.load("attributes_data.npy")

# Get unique values and their counts
unique_values, counts = np.unique(attribute_data, return_counts=True)
[unique_values,counts]


# In[11]:




# Define the chaos game representation (CGR) rules
def cgr_rules(amino_acid):
    if amino_acid == 'A':
        return 0.5, 0.5
    elif amino_acid == 'C':
        return 1.0, 0.5
    elif amino_acid == 'D':
        return 0.5, 1.0
    elif amino_acid == 'E':
        return 0.0, 0.5
    elif amino_acid == 'F':
        return 1.0, 1.0
    elif amino_acid == 'G':
        return 0.25, 0.25
    elif amino_acid == 'H':
        return 0.75, 0.25
    elif amino_acid == 'I':
        return 0.75, 0.75
    elif amino_acid == 'K':
        return 0.25, 0.75
    elif amino_acid == 'L':
        return 0.75, 0.0
    elif amino_acid == 'M':
        return 0.5, 0.0
    elif amino_acid == 'N':
        return 0.25, 0.5
    elif amino_acid == 'P':
        return 1.0, 0.0
    elif amino_acid == 'Q':
        return 0.0, 1.0
    elif amino_acid == 'R':
        return 0.5, 0.25
    elif amino_acid == 'S':
        return 0.75, 0.5
    elif amino_acid == 'T':
        return 0.5, 0.75
    elif amino_acid == 'V':
        return 0.0, 0.0
    elif amino_acid == 'W':
        return 1.0, 0.25
    elif amino_acid == 'Y':
        return 1.0, 0.75
    else:
        return 0.25, 1.0

# Define the recursive function to generate the kaleidoscope
def generate_kaleidoscope(protein_sequence, depth, position, angle, scale, color):
    
#     if depth <= 0:
#         return
    if depth <= 0:
        return
#     print("depth: ",depth)
    x, y = position
    dx = scale * np.cos(angle)
    dy = scale * np.sin(angle)
    
    for amino_acid in protein_sequence:
#         print(amino_acid)
        x, y = x + dx, y + dy
        cx, cy = cgr_rules(amino_acid)
        
        plt.plot([x, cx], [y, cy], color=color)
        plt.plot([x, cx], [y, -cy], color=color)
        plt.plot([-x, cx], [-y, cy], color=color)
        plt.plot([-x, cx], [-y, -cy], color=color)
        
        generate_kaleidoscope(protein_sequence, depth-1, (x, y), angle, scale, color)
        generate_kaleidoscope(protein_sequence, depth-1, (x, -y), angle, scale, color)
        generate_kaleidoscope(protein_sequence, depth-1, (-x, y), angle, scale, color)
        generate_kaleidoscope(protein_sequence, depth-1, (-x, -y), angle, scale, color)
        
        # Decrement depth here
        depth -= 1



# In[12]:


depth = 4  # Specify the recursion depth
position = (0, 0)  # Initial position of the central seed point
angle = 0  # Initial angle of rotation
scale = 10  # Scale factor for the replication
color = 'blue'  # Color for the replicated segments



images = []
labels = []

for i in range(len(seq_data)):
    print("index: ",i,"/",len(seq_data))
    sequence = seq_data[i]
    label = attribute_data[i]

    plt.figure(figsize=(8, 8))
    generate_kaleidoscope(sequence, depth, (0, 0), 0, scale, 'blue')
    plt.axis('off')

    image_path = f'{label}/{len(images)}.png'
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path)
    plt.close()

    images.append(image_path)
    labels.append(label)

# return images, labels


# In[13]:


# Example usage
protein_sequence = 'ACQRSTAGTACGT'  # Replace with your protein sequence
depth = 4  # Specify the recursion depth
position = (0, 0)  # Initial position of the central seed point
angle = 0  # Initial angle of rotation
scale = 10  # Scale factor for the replication
color = 'blue'  # Color for the replicated segments

# Generate the kaleidoscope image
plt.figure(figsize=(8, 8))
generate_kaleidoscope(protein_sequence, depth, position, angle, scale, color)
plt.axis('off')
plt.show()


# In[14]:


# Example usage
protein_sequence = 'AAAAAAAAAAAAA'  # Replace with your protein sequence
depth = 4  # Specify the recursion depth
position = (0, 0)  # Initial position of the central seed point
angle = 0  # Initial angle of rotation
scale = 10  # Scale factor for the replication
color = 'blue'  # Color for the replicated segments

# Generate the kaleidoscope image
plt.figure(figsize=(8, 8))
generate_kaleidoscope(protein_sequence, depth, position, angle, scale, color)
plt.axis('off')
plt.show()


# In[ ]:




