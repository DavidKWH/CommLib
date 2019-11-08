'''
high-level interface to google drive

Design:
    per user authentication setup
    look inside $HOME for application data
    includes:
        credentials.json
        token.pickle
'''
import os
import mimetypes
import pickle
from sys import exit

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from googleapiclient import errors
from googleapiclient.discovery import build
# file management
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaInMemoryUpload

################################################################################
# module initialization
################################################################################
# google drive root folder
rootdir = 'octopus'

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']

# look for user local data
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

def get_file(fname, parent):
    '''
    return file_id if it exists,
    otherwise return None
    '''
    assert parent, 'parent not set'
    not_trash_query = "trashed=false"
    folder_query = f"'{parent}' in parents"
    name_query = f"name='{fname}'"

    query = f"{folder_query} and {name_query} and {not_trash_query}"
    fields = 'nextPageToken, files(id, name)'

    response = service.files().list(q=query,
                                    spaces='drive',
                                    fields=fields).execute()

    for file in response.get('files', []):
        print (f"Found file: {file.get('name')}, {file.get('id')}")

    file_list = [ (file.get('name'), file. get('id'))
                    for file in response.get('files', []) ]

    if file_list:
        assert len(file_list) == 1, 'more than one file found'
        file_id = file_list[0][1]
        return file_id
    else:
        return None

def write_file(filepath, parent=None, mime_type=None, overwrite=True):
    '''
    write file to filepath, replace file if overwrite is True.

    assumptions:
        Infer mime type if not specified.
        filepath should be a valid file
        write to root directory if parent not given
    '''
    assert os.path.isfile(filepath), f'{filepath} must be a valid file'
    if not parent: parent='root'

    fname = os.path.basename(filepath)
    file_ext = os.path.splitext(fname)[1]

    # lookup mime type
    if not mime_type:
        mimetypes.init()
        mime_type = mimetypes.types_map.get(file_ext, 'application/octet-stream')

    metadata = {
        'name': fname,
    }

    media = MediaFileUpload(filepath, mimetype=mime_type, resumable=True)

    file_id = get_file(fname, parent)

    if file_id:
        print('File exists!')
        file = service.files().update(body=metadata,
                                      media_body=media,
                                      fileId=file_id).execute()
    else:
        metadata['parents'] = [parent]
        file = service.files().create(body=metadata,
                                      media_body=media,
                                      fields='id').execute()
        file_id = file.get('id')

    print(f'File ID: {file_id}')

    return file_id

def write_bytes(buf, filepath, parent=None, text=True, overwrite=True):
    '''
    write buf to filepath, replace file if overwrite is True

    assumptions:
        write to root directory if parent not given
    '''
    #assert os.path.isfile(filepath), f'{filepath} must be a valid file'
    if not parent: parent='root'

    fname = os.path.basename(filepath)
    #file_ext = os.path.splitext(fname)[1]

    # lookup mime type
    #mimetypes.init()
    #mime_type = mimetypes.types_map.get(file_ext, 'application/octet-stream')
    mime_type = 'text/plain' if text else 'application/octet-stream'

    metadata = {
        'name': fname,
    }

    media = MediaInMemoryUpload(buf, mimetype=mime_type, resumable=True)

    file_id = get_file(fname, parent)

    if file_id:
        print('File exists!')
        file = service.files().update(body=metadata,
                                      media_body=media,
                                      fileId=file_id).execute()
    else:
        metadata['parents'] = [parent]
        file = service.files().create(body=metadata,
                                      media_body=media,
                                      fields='id').execute()
        file_id = file.get('id')

    print(f'File ID: {file_id}')

    return file_id

################################################################################
# user API functions
################################################################################
def save_file(filepath, mimi_type=None):
    ''' upload file to google drive '''
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

def save_to_file(filepath, buf, text=True):
    ''' save buf to a remote file directly '''
    assert not os.path.isabs(filepath), 'support relative path only'
    filepath = os.path.join(rootdir, filepath)
    print(f'saving buffer to gdrive: {filepath}')

    basename = os.path.basename(filepath)
    dirname = os.path.dirname(filepath)
    dirname = os.path.normpath(dirname)

    dirs = split_all(dirname)
    print(dirs)
    folder = make_dirs(dirs)

    if text: buf = buf.encode()
    write_bytes(buf, basename, folder)


