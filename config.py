import os
import sys

def setup_paths():
    # Get the absolute path to the root directory
    root_dir = os.path.abspath(os.path.dirname(__file__))

    # Add directories to the Python path
    src_dir = os.path.join(root_dir, 'src')
    streamlit_app_dir = os.path.join(root_dir, 'streamlit_app')

    if src_dir not in sys.path:
        sys.path.append(src_dir)

    if streamlit_app_dir not in sys.path:
        sys.path.append(streamlit_app_dir)

# Call the setup_paths function to configure the paths
setup_paths()
