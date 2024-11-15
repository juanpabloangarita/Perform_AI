import os

def get_full_path(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located i.e. data_loader
    dir_script_dir = os.path.dirname(script_dir)  # Get the directory where the previous dir is located i.e. src
    project_dir = os.path.dirname(dir_script_dir)  # Get the directory where the previous dir is located i.e. Perform_AI

    full_path = os.path.join(project_dir, file_path)
    return full_path
