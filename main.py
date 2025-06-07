from pipelines.feature_pipeline import main as feature_main
from pipelines.train_pipeline import main as train_main
from pipelines.inference_pipeline import main as inference_main

# Defining the callable function to execute ALL three pipelines.
def main():
    print("Running ML full Pipelines...")

    feature_main()
    train_main()
    inference_main()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ML pipeline stages...")
    parser.add_argument("--run", choices=['all', 'feature', 'train', 'infer'], default='all', help="Choose what part of the pipeline to run...")
    args = parser.parse_args()

    if args.run == 'feature':
        feature_main()
    elif args.run == 'train':
        train_main()
    elif args.run == 'infer':
        inference_main()
    else:
        main() 
    