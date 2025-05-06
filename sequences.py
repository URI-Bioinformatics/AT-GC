import os
from Bio import SeqIO

# For debugging purposes
files = 0
seq_count = 0
max_length = 0


"""Reads a .fna file and returns the first sequence in the file."""
def read_fna(file_path):
    global seq_count, max_length

    with open(file_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):

            # Read sequence description
            # description = [word for word in record.description.split(" ")]
            # genus = description[1]
            
            # Finding maximum length for padding
            """
            if (max_length < len(record.seq)): 
                max_length = len(record.seq)
            """
            
            # We only care for the top sequence
            seq_count += 1
            
            return str(record.seq)



"""
A module iterates over fna data files that hold DNA sequences sorted in folders of phylums.
The program loads the data into a dictionary that uses genises as keys and lists of 
k-mers of those sequeces as values.
This dictionary will be split and used to train a transformers ML model and then test it 
on its ability to predict the genis of a genome sucessfully
"""
def genome_tuples():
    global files, seq_count, max_length

    print(f"Collected data..")
    # Hardcoded data source
    # This data folder holds genomic data sorted in phylum folders
    #phylum_data_folder_path = "/scratch/class/ayman_sandouk_uri_edu-bps542/Project/phylum_data"
    phylum_data_folder_path = "Data"



    # Create an empty phylum list 
    phylum_seq = []
    
    num_of_phylums = len(os.listdir(phylum_data_folder_path))
    # Start iterating over the folders
    for phylum in os.listdir(phylum_data_folder_path):

        phylum_folder = os.path.join(phylum_data_folder_path, phylum)

        # Just counting fna files. Should have the same number of sequences 
        number_of_files= len(os.listdir(phylum_folder)) - 3     # We have 3 metadata files in each folder
        
        # Debugging
        print(f"In {phylum} folder, we have {number_of_files} data files.") 
        files += number_of_files

        # Each directory holds a single fna file
        for directory in os.listdir(phylum_folder):
            data_directory_path = os.path.join(phylum_folder, directory)

            # Skipping metadata files
            if os.path.isdir(data_directory_path):

                data_file = os.listdir(data_directory_path)

                # Confirm that there's only one file in each folder
                assert len(data_file) == 1

                data_file_path = os.path.join(data_directory_path, data_file[0])
                if os.path.isfile(data_file_path): # Just for more safty
                    # Get the top sequence in the data file as a string, map it as an item of a tuple with its phylum
                    
                    sequence= read_fna(data_file_path)
                    phylum_seq.append((sequence, phylum))



    print(f"Done collecting data..\n")


    # Debugging
    """
    # For testing purposes: Outputting the inputted data
    # Iterate over the dictionary items 
    for sequences, phylum in phylum_seq:
        print(f"{phylum} - {sequence[:70]}"+"..." if len(sequence) > 70 else f"Sequence: {sequence}")

    print("Datatypes:")
    print(type(phylum_seq))
    print(type(phylum_seq[0]))
    print(type(phylum_seq[0][0]))
    print(type(phylum_seq[0][1]))

    
    # I've noiticed that some files have more than one sequence in them. This is to confirm that
    # AS: Confirmed! Some files have more than one sequence of the same species!!!
    print(f"Total files: {files}")
    print(f"Total sequences: {seq_count}")
    print(f"Total # of pyhlums: {num_of_phylums}")
    print(f"Total # of tuples: {len(phylum_seq)}")
    print(f"Max seq. length: {max_length}")
    """
    return phylum_seq