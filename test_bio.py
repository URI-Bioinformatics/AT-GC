import os
from Bio import SeqIO
# from Bio import LogisticRegression


def read_fna(file_path):
    """Reads a .fna file and returns a list of SeqRecord objects."""
    records = []
    with open(file_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            records.append(record)
    return records

# Example usage
sequence_iterators = {}
data_folder_path = "../data"
for directory in os.listdir(data_folder_path):
    if directory != ".gitkeep" and directory != "dataset_catalog.json" and directory != "data_summary.tsv"  and directory != "assembly_data_report.jsonl" :
        data_directory_path = os.path.join(data_folder_path, directory)
        data_file = os.listdir(data_directory_path)
        data_file_path = os.path.join(data_directory_path, data_file[0])
        if os.path.isfile(data_file_path):
            # print(data_file_path)
            sequence_iterators[data_file[0]] = (read_fna(data_file_path))
print(f"Collected data..\n{len(sequence_iterators)}")
print()
for file, sequences in sequence_iterators.items():
    print(f"Info from {file}")

    for record in sequences:
        print(f"ID: {record.id}")
        print(f"Description: {record.description}")
        print(f"Sequence: {record.seq[:70]}"+"..." if len(record.seq) > 70 else f"Sequence: {record.seq}")
        print("---")