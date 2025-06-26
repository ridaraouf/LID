import os
import shutil

def flatten_split(split_dir: str, target_dir: str):
    """
    Flattens a given split ('train' or 'test') by collecting all language folders
    from each sub-dataset and moving them into a common structure based on language.
    """
    os.makedirs(target_dir, exist_ok=True)

    for dataset_name in os.listdir(split_dir):
        dataset_path = os.path.join(split_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue  # Skip files

        for lang_name in os.listdir(dataset_path):
            lang_path = os.path.join(dataset_path, lang_name)
            if not os.path.isdir(lang_path):
                continue

            target_lang_path = os.path.join(target_dir, lang_name)
            os.makedirs(target_lang_path, exist_ok=True)

            for file_name in os.listdir(lang_path):
                src_file = os.path.join(lang_path, file_name)

                # Add dataset prefix to prevent collisions
                new_file_name = f"{dataset_name}_{file_name}"
                dst_file = os.path.join(target_lang_path, new_file_name)

                shutil.copy2(src_file, dst_file)  # Use move() if you want to delete originals

def main():
    root_dir = '/home/teaching/conformer_mfcc/split_dataset'  # Make sure this is the correct path
    test_old = os.path.join(root_dir, 'test')
    train_old = os.path.join(root_dir, 'train')

    test_new = os.path.join(root_dir, 'test_pre')
    train_new = os.path.join(root_dir, 'train_pre')

    print("Flattening test split...")
    flatten_split(test_old, test_new)

    print("Flattening train split...")
    flatten_split(train_old, train_new)

    print("✅ Dataset structure flattened successfully!")

if __name__ == '__main__':
    main()
