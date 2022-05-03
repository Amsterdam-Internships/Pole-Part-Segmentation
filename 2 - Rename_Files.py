# Python3 code to rename multiple
# files in a directory or folder

# importing os module
import os


# Function to rename multiple files
def main():
    base_path = r'/home/sara/Desktop/Internship/Experiment Results//Data/Set3_1'
    folders = ['modified_val_ply','modified_test_ply', 'modified_train_ply']
    for folder in folders:
        folder = os.path.join(base_path,folder)
        i=0
        for count, filename in enumerate(os.listdir(folder)):
            dst = f"Pole_00{(str(count).zfill(2))}.ply"
            src = f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
            dst = f"{folder}/{dst}"

            # rename() function will
            # rename all the files
            os.rename(src, dst)



# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()