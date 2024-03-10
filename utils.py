import os

def load_certs(cabundle_file :str = "cabundle.pem"):
    if os.path.isfile(cabundle_file):
        os.environ['TOOL_CONDA_SSL_VERIFY'] = cabundle_file
        os.environ['GIT_SSL_CAINFO'] = cabundle_file
        os.environ['SSL_CERT_FILE'] = cabundle_file
        os.environ['REQUESTS_CA_BUNDLE'] = cabundle_file
