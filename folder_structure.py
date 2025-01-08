import os

def print_folder_structure(path, indent=0, max_depth=3):
    if not os.path.exists(path):
        print(f"Path '{path}' does not exist.")
        return

    # Only proceed if the current depth is less than the maximum depth
    if indent > max_depth:
        return

    # List the items in the specified directory
    items = os.listdir(path)
    for item in items:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            print('    ' * indent + '|-' + item)
            print_folder_structure(item_path, indent + 1, max_depth)

# Example usage
folder_path = r"C:\dataset\vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0\data"
print_folder_structure(folder_path)
