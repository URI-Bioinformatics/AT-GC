import os
from Bio import SeqIO
# from Bio import LogisticRegression

# For debugging purposes
files = 0
seq_count = 0

"""Reads a .fna file and returns a list of SeqRecord objects."""
def read_fna(file_path):
    global seq_count
    records = []
    with open(file_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            records.append(record)
        # records = SeqIO.parse(handle, "fasta") 
    seq_count += len(records)
    return records


"""
A module iterates over fna data files that hold DNA sequences sorted in folders of phylums.
The program loads the data into a dictionary that uses genises as keys and lists of 
k-mers of those sequeces as values.
This dictionary will be split and used to train a transformers ML model and then test it 
on its ability to predict the genis of a genome sucessfully
"""
if __name__ == "__main__":
    

    # Hardcoded data source
    # This data folder holds genomic data sorted in phylum folders
    phylum_data_folder_path = "/scratch/class/ayman_sandouk_uri_edu-bps542/Project/phylum_data"
    # This data folder holds unsorted genomic data
    data_folder_path = "/scratch/class/ayman_sandouk_uri_edu-bps542/Project/data"


    # Create empty dictionaries 
    phylum_dict = {}
    genis_dict = {}
    
    # Start iterating over the folders
    for phylum in os.listdir(phylum_data_folder_path):
        # Create pyhlum keys with values that are empty lists
        phylum_dict[phylum] = []

        phylum_folder = os.path.join(phylum_data_folder_path, phylum)
        number_of_files= len(os.listdir(phylum_folder)) - 3     # We have 3 metadata files in each folder
        
        # Debugging
        print(f"In {phylum} folder, we have {number_of_files} data files.") 
        files += number_of_files
        # Limit the work to 5 sequeces from each phylum to test
        num = 0 # Debugging: Testing. Comment out after confirming


        # Each directory holds a single fna file
        for directory in os.listdir(phylum_folder):
            if num < 5: # Debugging: Testing. Comment out after confirming
                data_directory_path = os.path.join(phylum_folder, directory)
                if os.path.isdir(data_directory_path):
                    data_file = os.listdir(data_directory_path)

                    # Confirm that there's only one file in each folder
                    assert len(data_file) == 1
                    data_file_path = os.path.join(data_directory_path, data_file[0])
                    if os.path.isfile(data_file_path): # Just for more safty
                        # print(data_file_path)
                        # Get the content of the data file in a bio object, map it as an item of a list of object in its respective phylum key
                        phylum_dict[phylum] += read_fna(data_file_path)
            else: break # Debugging: Testing. Comment out after confirming
            num += 1    # Debugging: Testing. Comment out after confirming

    print(f"Collected data..\n")

    # For testing purposes: Outputting the inputted data
    # Iterate over the dictionary items 
    for phylum, sequences in phylum_dict.items():
        print(f"Info from {phylum}")
        print(f"This phylum contains {len(sequences)} sequence objects.")
        for record in sequences:
            print(f"ID: {record.id}")
            print(f"Description: {record.description}")
            print(f"Sequence len: {len(record.seq)}")
            print(f"Sequence: {record.seq[:70]}"+"..." if len(record.seq) > 70 else f"Sequence: {record.seq}")
            print("---")
    # I've noiticed that some files have more than one sequence in them. This is to confirm that
    # AS: Confirmed! Some files have more than one sequence of the same species!!!
    print(f"Total sequences: {seq_count}")
    print(f"Total files: {files}")
    
