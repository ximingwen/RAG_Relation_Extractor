This folder contains various RE techniques:
1. RE Classifier
2. Data Augmentation with GPT-2
3. Distant Supervision

The following are the folders/files contained in RE_Model and their purpose:

1. all_results

	This folder contains all of the test results performed during the Data Augmentation w/ GPT2 test.

	all_results
		|
		|__results_full (contains test results done with (ChemProt.csv)
		|	|
		|	|
		|	|__results (contains test results without using data augmentation)
		|	|
		|	|__results_dare (contains test results with using data augmentation)
		|
		|
		|
		|__results_reduced (contains test results done with ChemProt_Reduced.csv)
                        |
                        |
                        |__results (contains test results without using data augmentation)
                        |
                        |__results_dare (contains test results with using data augmentation)

2. config_files

	This folder contains 5 files that were each used to configure the RE classifier and the GPT2 (DARE) model. 
	
	label_map.json - Maps original 13 unique ChemProt labels to the 5 unique labels
	labels.txt - Names of the 5 unique labels
	paths.json - All paths needed during DARE testing (huggingface APIs, local model/data paths…)
	re_config - Configurations for classifier (model hyper-parameters...)
	dare_config - Configurations for GPT2 model (model hyper-parameters...)

3. data

	ChemProt.csv - Full ChemProt dataset
	ChemProt_Reduced.csv - Reduced ChemProt dataset
	**Other CSVs in this folder are temporoary files that are created during each run of the dare test (run_tests.py).
	    Synthetic_ChemProt.csv, train_dare_data.csv, and test_dare_data.csv are the locations that the DARE test writes files during so that the two RE classifiers can train and test on.

4. models
	
	Where the huggingface models and tokenizers are saved. This simply prevents excessive API calls to huggingface during testing. These are large models and are not necessary to keep stored, however the run_tests.py file will save models during the tests in those locations.

5. modules

	Folder containing python modules. 

	utils.py - Contains simple utility functions necessary during DARE testing
	models.py - Contains 4 python classes used during DARE testing.
		
		4 Classes in models.py:
		1. MyDataset - Inherits the PyTorch Dataset class. Has len() and get_item() methods which are necessary to utilize PyTorch’s DataLoader for batch training.
		2. MyModule - Takes in dataset and splits data into train and test. Has method to retrieve a DataLoader of the training data and DataLoader of testing data.
		3. Data_Augmenter - Holds the GPT2 model and tokenizer. Facilitates the fine-tuning and text generation process as well as other minor details like saving new synthetic dataset.
		4. RE_Classifier - Holds Biomed RoBERTa model. Facilitates the fine-tuning and eval process.

6. run_tests.py
	
	This file runs the DARE tests. Example running 100 tests:
	
	`python3 run_tests.py --n_tests 100 --reduced_data --train_dare`
	
	The n_tests specifies for how many tests should be run.
	The reduced_data flag signals to use ChemProt_Reduced.csv rather than the full ChemProt.csv
	The train_dare flag signals whether or not to use data augmentation during tests.

7. Distant_Supervised

	This folder contains all of the information regarding an unsupervised framework to classify relationships within the ChemProt dataset. The folder, Word2VecTools, has the pre-trained BioASQ word vectors. The python notebook, distant_supervision.ipynb is the code that actually carries out the unsupervised technique.

