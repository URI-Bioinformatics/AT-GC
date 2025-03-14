import os
from Bio import SeqIO

def read_fna(file_path):
    """Reads a .fna file and returns a list of SeqRecord objects."""
    records = []
    with open(file_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            records.append(record)
    return records

# Example usage
sequence_iterators = {}
data_folder_path = "test_data"
for file in os.listdir(data_folder_path):
    file_path = os.path.join(data_folder_path, file)
    if os.path.isfile(file_path):
        sequence_iterators[file] = (read_fna(file_path))

for file, sequences in sequence_iterators.items():
    print(f"Info from {file}")
for record in sequences:
    print(f"ID: {record.id}")
    print(f"Sequence: {record.seq}")
    print(f"Description: {record.description}")
    print("---")