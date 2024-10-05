import argparse
from modules.models import RE_Classifier, Data_Augmenter
from modules.util import swap_dataset
from tqdm.auto import tqdm


if __name__=="__main__":
    # Get arguments
    parser = argparse.ArgumentParser(description="Arguments for RE baseline tests.")
    parser.add_argument("--gpu", default=0, type=int, help="Index of GPU to use.")
    parser.add_argument("--n_tests", default=1, type=int, help="Number of times to repeat the script.")
    parser.add_argument("--train_dare", action="store_true", help="Flag uses data augmentation to increase train data.")
    parser.add_argument("--reduced_data", action="store_true", help="Use reduced ChemProt dataset.")
    parser.add_argument("--start_from", default=0, type=int, help="Number test to start at.")
    args = parser.parse_args()
    GPU_DEVICE = args.gpu
    NUM_TESTS = args.n_tests
    TRAIN_DARE = args.train_dare
    USE_REDUCED = args.reduced_data
    START_TEST = args.start_from

    if USE_REDUCED:
        swap_dataset("data/ChemProt_Reduced.csv")
    else:
        swap_dataset("data/ChemProt.csv")
    
    tests_left = NUM_TESTS - START_TEST 
    progress_bar = tqdm(range(tests_left), desc="All Tests")

    for i in range(START_TEST, NUM_TESTS):

        if TRAIN_DARE:
            # Train/Generate new sentences

            dare_model = Data_Augmenter(gpu_device=GPU_DEVICE)

            for id in range(5):

                dare_model.generate(id)

            # Save examples
            dare_model.save_examples()
            # Combine examples
            dare_model.save_train_test()
            
            classifier = RE_Classifier(gpu_device=GPU_DEVICE, loaded_data=TRAIN_DARE, strip_fakes=False)
            print("trainig")
            classifier.train()
            classifier.test()
            output_path = "all_results/"
            output_path += "results_reduced/" if USE_REDUCED else "results_full/"
            output_path += "results_dare/"
            output_path += f"results_{i}.json"
            classifier.save_results(output_path)
        print("here")
        # Classifier Part
        classifier = RE_Classifier(gpu_device=GPU_DEVICE, loaded_data=True, strip_fakes=True)
        # Train Model
        classifier.train()
        # Test Model
        classifier.test()
        # Save Results
        output_path = "all_results/"
        output_path += "results_reduced/" if USE_REDUCED else "results_full/"
        output_path += "results/"
        output_path += f"results_{i}.json"
        classifier.save_results(output_path)

        progress_bar.update(1)

    exit(0)
