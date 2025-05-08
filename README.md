# AT-GC

Accurate Taxonomic Genomic Computer

This research is about classifying microbes into phyla by genomic signatures using machine learning


## Usage
- To run this program start you can start by downloading the repo manually or cloning it using the your favorite terminal and the `git` tool:

```bash
git clone https://github.com/URI-Bioinformatics/AT-GC.git
```

- The next step is to set up your python environment to be able to run the code. We do so by installing the required packages found on `requirements.txt`

  
```
cd AT-GC
pip install -r requirements.txt
```

- Next, make sure you download your data and make sure the data is organized as such:
```
Data/
├──Cyanobacteriota/
|   |── assembly_data_report.jsonl
|   ├──── dataset_catalog.json
|   ├──── data_summary.tsv
|   ├──── GCF_000011345.1
|   |     └── GCF_000011345.1_ASM1134v1_genomic.fna
|   ├──── GCF_000011385.1
|   |     └── GCF_000011385.1_ASM1138v1_genomic.fna
|   |     .
|   |     .
|   |     .
├──Fusobacteriota/
|   |── assembly_data_report.jsonl
|   ├──── dataset_catalog.json
|   ├──── data_summary.tsv
|   ├──── GCF_000011355.1
|   |     └── GCF_000011355.1_ASM1134v1_genomic.fna
```

- Run the program passing the path to the data folder (full or relative as an argument
```
python model_script.py /path/to/data/folder
```

- Finally, results can be found in `results_summary.txt` and individual validation graphs can be found in individual k-mers folders `results_k{k}`
