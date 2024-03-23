import argparse
import os
import urllib.request


def download_files(download_links, save_folder):
    """
    Download files from a list of given download links and save them to a specified folder.

    Parameters:
    - download_links (list): A list of URLs pointing to the files to be downloaded.
    - save_folder (str): The local directory where the downloaded files will be saved.

    Returns:
    None

    Example:
    ```python
    download_links = ["https://example.com/file1.txt", "https://example.com/file2.jpg"]
    save_folder = "/path/to/save/folder"
    download_files(download_links, save_folder)
    ```
    """
    print(f"Downloading {len(download_links)} files from loghub:")
    for link in download_links:
        file_name = os.path.join(save_folder, link.split("/")[-1].split("?")[0])
        urllib.request.urlretrieve(link, file_name)
        print(f"Downloaded: {file_name}")


def unzip_files(save_folder):
    """
    Unzip files in a specified folder. Supports both ZIP and tar.gz formats.
    Deletes the compressed files after decompressing.

    Parameters:
    - save_folder (str): The local directory containing the files to be unzipped.

    Returns:
    None

    Example:
    ```python
    save_folder = "/path/to/save/folder"
    unzip_files(save_folder)
    ```
    """
    for file in os.listdir(save_folder):
        print(f"Unzipping: {file}")
        unzip_dir = os.path.join(save_folder, file.split('.')[0])

        if not os.path.exists(unzip_dir):
            os.makedirs(unzip_dir)
        
        
        if ".zip" in file:
            os.system(f"unzip {os.path.join(save_folder, file)} -d {unzip_dir}")
        elif ".tar.gz" in file:
            os.system(f"tar -xf {os.path.join(save_folder, file)} -C {unzip_dir}")

        os.remove(os.path.join(save_folder, file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='loghub_downloader',
        description='This script downloads each dataset from the Loghub project and unzips them into the save_folder. This script prepares the data to be preprocessed by data_preprocess.py',
    )
    parser.add_argument('-s', '--save_folder', required=True) 
    args = parser.parse_args()

    # Create the save folder if it doesn't exist
    if not os.path.exists(args.save_folder):
        print(f"Creating {args.save_folder} folder as it doesn't exist...")
        os.makedirs(args.save_folder)

    download_links = [
        "https://zenodo.org/records/8196385/files/Android_v1.zip?download=1",
        "https://zenodo.org/records/8196385/files/Apache.tar.gz?download=1",
        "https://zenodo.org/records/8196385/files/BGL.zip?download=1",
        "https://zenodo.org/records/8196385/files/Hadoop.zip?download=1",
        "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1",
        "https://zenodo.org/records/8196385/files/HealthApp.tar.gz?download=1",
        "https://zenodo.org/records/8196385/files/HPC.zip?download=1",
        "https://zenodo.org/records/8196385/files/Linux.tar.gz?download=1",
        "https://zenodo.org/records/8196385/files/Mac.tar.gz?download=1",
        "https://zenodo.org/records/8196385/files/OpenStack.tar.gz?download=1",
        "https://zenodo.org/records/8196385/files/Proxifier.tar.gz?download=1",
        "https://zenodo.org/records/8196385/files/Spark.tar.gz?download=1",
        "https://zenodo.org/records/8196385/files/SSH.tar.gz?download=1",
        "https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1",
        "https://zenodo.org/records/8196385/files/Windows.tar.gz?download=1",
        "https://zenodo.org/records/8196385/files/Zookeeper.tar.gz?download=1",
        "https://zenodo.org/records/8196385/files/HDFS_v2.zip?download=1"
    ]

    download_files(download_links=download_links, save_folder=args.save_folder)

    unzip_files(save_folder=args.save_folder)
