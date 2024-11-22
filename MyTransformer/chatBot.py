# chatBot.py
import processMovieCorpusDataset  # Ensure this is the correct module name
# runDatasetDownload.py
import datasetDownloader  # Ensure this is the correct module name
if __name__ == "__main__":
    config_file = 'config.json'  # Path to your config file
    datasetDownloader.process_dataset(config_file)
    processMovieCorpusDataset.process_dataset(config_file)
    