import os
from typing import Dict


def count_class_occurrences(
    folder_path: str, class_names: Dict[int, str]
) -> Dict[str, int]:
    """
    Count the number of occurrences of each class in the dataset.

    Args:
        folder_path (str): Path to the folder containing the txt files.
        class_names (Dict[int, str]): Dictionary mapping class IDs to class names.

    Returns:
        Dict[str, int]: A dictionary containing the count of occurrences for each class name.
    """
    # Initialize a dictionary to count occurrences of each class ID
    class_counts = {class_id: 0 for class_id in class_names.keys()}

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1

    # Map class IDs to class names and return the result
    class_occurrences = {
        class_names[class_id]: count for class_id, count in class_counts.items()
    }
    return class_occurrences


def remove_empty_txt_files(directory_path: str) -> None:
    """
    Remove the txt files in a folder if they are empty.

    Args:
        directory_path (str): Path to the directory containing the txt files.
    """
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        # Construct full file path
        file_path = os.path.join(directory_path, filename)

        # Check if the file is a txt file and if it is empty
        if (
            filename.endswith(".txt")
            and os.path.isfile(file_path)
            and os.path.getsize(file_path) == 0
        ):
            # If the file is empty, remove it
            os.remove(file_path)
            print(f"Removed empty file: {filename}")


def change_class_ids_in_file(file_path: str, class_mapping: dict) -> None:
    """
    Change class IDs in a file according to the provided mapping.

    Args:
        file_path (str): Path to the file to be modified.
        class_mapping (dict): Dictionary mapping old class IDs to new class IDs.
    """
    with open(file_path, "r+") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) > 0:
                old_class_id = int(parts[0])
                if old_class_id in class_mapping:
                    parts[0] = str(class_mapping[old_class_id])
                    lines[i] = " ".join(parts) + "\n"
        file.seek(0)
        file.truncate()
        file.writelines(lines)


def change_class_ids_in_directory(directory_path: str, class_mapping: dict) -> None:
    """
    Change class IDs in all files within a directory according to the provided mapping.

    Args:
        directory_path (str): Path to the directory containing the files to be modified.
        class_mapping (dict): Dictionary mapping old class IDs to new class IDs.
    """
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                change_class_ids_in_file(file_path, class_mapping)
