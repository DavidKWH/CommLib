'''
high-level interface to google drive 

Design:
    per user authentication setup
    look inside $HOME for application data
    includes:
        credentials.json
        token.pickle
'''
import pickle
import os
from sys import exit

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
# file management
from googleapiclient.http import MediaFileUpload

################################################################################
# module initialization
################################################################################
# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']

# look for user local data
rootdir = 'octopus'
dirname = os.path.join(os.environ['HOME'], '.octopus')

token_name = 'token.pickle'
token_name = os.path.join(dirname, token_name)
cred_name = 'credentials.json'
cred_name = os.path.join(dirname, cred_name)

assert os.path.exists(cred_name), f'missing credentials in {dirname}'

creds = None
# The file token.pickle stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first
# time.
if os.path.exists(token_name):
    with open(token_name, 'rb') as token:
        creds = pickle.load(token)

# If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        # refresh token
        creds.refresh(Request())
    else:
        # send user to log in screen
        flow = InstalledAppFlow.from_client_secrets_file(
            cred_name, SCOPES)
        creds = flow.run_local_server(port=0)
    # save the credentials for the next run
    with open(token_name, 'wb') as token:
        pickle.dump(creds, token)

# create service
service = build('drive', 'v3', credentials=creds)

################################################################################
# support functions
################################################################################
def split_all(path):
    dirs = []
    while True:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            dirs.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            dirs.insert(0, parts[1])
            break
        else:
            path = parts[0]
            dirs.insert(0, parts[1])
    return dirs

################################################################################
# functions
################################################################################
def get_folder(fname, parent):
    ''' 
    check if folder exists, if so returns folder id,
    else return None
    '''
    folders_only_query = "mimeType='application/vnd.google-apps.folder'"
    not_trash_query = "trashed=false"
    root_folder_query = f"'{parent}' in parents"
    folder_name_query = f"name='{fname}'"

    query = f"{root_folder_query} and {folders_only_query} and {folder_name_query} and {not_trash_query}"
    fields = 'nextPageToken, files(id, name)'

    response = service.files().list(q=query,
                                    spaces='drive',
                                    fields=fields).execute()

    for file in response.get('files', []):
        print (f"Found folder: {file.get('name')}, {file.get('id')}")

    file_list = [ (file.get('name'), file. get('id'))
                    for file in response.get('files', []) ]

    if file_list:
        assert len(file_list) == 1, 'more than one folder found'
        folder_id = file_list[0][1]
        return folder_id
    else:
        return None

def make_dirs(fnames, exist_ok=True):
    ''' 
    make all folders specified in folders
    if they do not already exist
    '''
    parent = 'root'
    for fname in fnames:
        folder = get_folder(fname, parent)
        if not folder:
            folder = create_folder(fname, parent)
        # folder is valid at this point
        parent = folder

    # return folder id
    return folder

def create_folder(fname, parent):
    ''' create folder '''

    file_metadata = {
        'name': fname,
        'parents': [parent],
        'mimeType': 'application/vnd.google-apps.folder',
    }

    file = service.files().create(body=file_metadata,
                                  fields='id').execute()

    folder_id = file.get('id')
    print(f"Folder ID: {folder_id}")

    return folder_id

def write_file(filepath, parent, text=False):

    fname = os.path.basename(filepath)
    mime_type = 'text/plain' if text else 'application/octet-stream'

    file_metadata = {
        'name': fname,
        'parents': [parent]
    }

    media = MediaFileUpload(filepath,
                            mimetype=mime_type,
                            resumable=True)

    file = service.files().create(body=file_metadata,
                                  media_body=media,
                                  fields='id').execute()

    file_id = file.get('id')
    print(f'File ID: {file_id}')

################################################################################
# user API functions
################################################################################
def save_file(filepath, text=False):
    ''' user interface for writing to google drive '''
    assert not os.path.isabs(filepath), 'support relative path only'
    filepath = os.path.join(rootdir, filepath)
    print(f'saving file to gdrive: {filepath}')

    basename = os.path.basename(filepath)
    dirname = os.path.dirname(filepath)
    dirname = os.path.normpath(dirname)

    dirs = split_all(dirname)
    print(dirs)
    folder = make_dirs(dirs)
    write_file(basename, folder)

