# Analyzing Dataset Annotation Quality Management in the Wild 	

### Jan-Christoph Klie, Richard Eckart de Castilho and Iryna Gurevych
#### [UKP Lab, TU Darmstadt](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp)

Source code for our experiments of our [arXiv paper](https://arxiv.org/abs/2307.08153).

> **Abstract:** Data quality is crucial for training accurate, unbiased, and trustworthy machine learning models and their correct evaluation. Recent works, however, have shown that even popular datasets used to train and evaluate state-of-the-art models contain a non-negligible amount of erroneous annotations, bias or annotation artifacts. There exist best practices and guidelines regarding annotation projects. But to the best of our knowledge, no large-scale analysis has been performed as of yet on how quality management is actually conducted when creating natural language datasets and whether these recommendations are followed. Therefore, we first survey and summarize recommended quality management practices for dataset creation as described in the literature and provide suggestions on how to apply them. Then, we compile a corpus of 591 scientific publications introducing text datasets and annotate it for quality-related aspects, such as annotator management, agreement, adjudication or data validation. Using these annotations, we then analyze how quality management is conducted in practice. We find that a majority of the annotated publications apply good or very good quality management. However, we deem the effort of 30% of the works as only subpar. Our analysis also shows common errors, especially with using inter-annotator agreement and computing annotation error rates.

* **Contact person:** Jan-Christoph Klie, ukp@mrklie.com
    * UKP Lab: http://www.ukp.tu-darmstadt.de/
    * TU Darmstadt: http://www.tu-darmstadt.de/

Drop me a line or report an issue if something is broken (and shouldn't be) or if you have any questions.

For license information, please see the `LICENSE` and `README` files.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## Project structure

This repository contains the code to download and pre-process the publications which are annotated for 
analyzing quality management as well as to mine the results.
Publication info is converted to a format [INCEpTION](https://github.com/inception-project/inception) can import.
INCEpTION then is used as the annotation tool.
Annotations are exported are further processed with code from this repository.

## Requirements

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Data

The data and results can be downloaded from [here](https://doi.org/10.48328/tudatalib-1220).

## Annotation Process

### 1. Finding Papers To Annotate

We use a snapshot of the [Papers With Code](https://github.com/paperswithcode/paperswithcode-data) data from the 26th of
November 2022. From that, we select the *text* datasets and match them against the 
[ACL Anthology](https://github.com/acl-org/acl-anthology) with the commit `3e0966ac`. While the ACL Anthology also 
contains backlinks to *Papers With Code*, they were still very few (~100 datasets marked at the time of writing). 
Hence, we opted for manually matching them by title.

| **File Name**                         | **md5**                          |
|:--------------------------------------|:---------------------------------|
| datasets.json.gz                      | 57193271ad26d827da3666e54e3c59dc |
| papers-with-abstracts.json.gz         | 4531a8b4bfbe449d2a9b87cc6a4869b5 |
| links-between-papers-and-code.json.gz | 424f1b2530184d3336cc497db2f965b2 |

In order to crawl the papers from the *Paper With Code* snapshot, put the files from the table above to
`data/external/paperswithcode`, checkout the ACL anthology to `data/external/acl-anthology` and then run 
`scripts/select_papers_with_code_dataset_papers.py`. This might take a while.

### 2. Prepare Files for INCEpTION

We use [INCEpTION](https://inception-project.github.io/) to annotate the PDFs for their quality management.
Annotation was made in batches. Create these by running `scripts/prepare_inception_projects.py`.
Create a project and import the layers from the accompanying data and import the PDFs.
Then you are ready to annotate!

### 3. Download Annotations from INCEpTION

Run `scripts/get_annotations_from_inception.py` to download the data from INCEpTION.
Then run `scripts/convert_annotations_from_jsoncas.py` to convert it to a more usable format and `scripts/clean_data.py`
to make it more consistent.

### 4. Analyze Data

Run `scripts/analyze_data.py` to get the counts and percentages reported in this work.

## Citing

Please use the following citation:

```
@misc{klie2023analyzing,
      title={Analyzing Dataset Annotation Quality Management in the Wild}, 
      author={Jan-Christoph Klie and Richard Eckart de Castilho and Iryna Gurevych},
      year={2023},
      eprint={2307.08153},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```