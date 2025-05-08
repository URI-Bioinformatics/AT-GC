"""Reads a .fna file and returns a list of SeqRecord objects."""
def read_fna(file_path):
    global seq_count
    records = []
    with open(file_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            records.append(record)
            print(dir(record))
            print()
            print(type(record))
            print(record.description)
            for word in record.description.split(" "):
                print(word)
            print(record.annotations)
            print(record.dbxrefs)
            print(record.features)
            print(record.letter_annotations)
            print(record.id)
            # print(record.translate()) # Method to turn into protein??
            # print(record.count()) # Method requires arg:sub
            # print(record.format()) # Method requires arg:format
            # print(record.isupper()) # Method base case
            # print(record.upper())  # Method base case
            # print(record.islower()) # Method base case
            # print(record.lower()) # Method base case
            exit()
        # records = SeqIO.parse(handle, "fasta") 
    seq_count += len(records)
    return records