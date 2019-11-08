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
import glob
import mimetypes
import pickle
from sys import exit

from googleapiclient import errors
from googleapiclient.discovery import build
# file management
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaInMemoryUpload
# oauth related
from httplib2 import Http
from oauth2client import file, client, tools

def get_authenticated(SCOPES, credential_file='credentials.json',
                  token_file='token.json', service_name='drive',
                  api_version='v3'):
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    store = file.Storage(token_file)
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets(credential_file, SCOPES)
        creds = tools.run_flow(flow, store)
    service = build(service_name, api_version, http=creds.authorize(Http()))
    return service

################################################################################
# module initialization
################################################################################
# initialize mimetypes
mimetypes.init()
mimetypes.add_type('application/python-pickle', '.pickle')
mimetypes.add_type('application/numpy-npy', '.npy')
mimetypes.add_type('application/numpy-npz', '.npz')

# google drive root folder
rootdir = 'octopus'

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']

# look for user local data
dirname = os.path.join(os.environ['HOME'], '.octopus')

token_name = 'token.json'
token_name = os.path.join(dirname, token_name)
cred_name = 'credentials.json'
cred_name = os.path.join(dirname, cred_name)

assert os.path.exists(cred_name), f'missing credentials in {dirname}'
#assert os.path.exists(token_name), f'missing token storage in {dirname}'

service = get_authenticated(SCOPES,
                            credential_file=cred_name,
                            token_file=token_name)

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

def create_file(filepath, parent=None, mime_type=None):
    '''
    create file at filepath

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
        mime_type = mimetypes.types_map.get(file_ext, 'application/octet-stream')

    metadata = {}
    metadata['name'] = fname
    metadata['parents'] = [parent]

    media = MediaFileUpload(filepath, mimetype=mime_type, resumable=True)

    file = service.files().create(body=metadata,
                                  media_body=media,
                                  fields='id').execute()
    print(f'File ID: {file["id"]}')
    return file['id']

def update_file(filepath, file_id, mime_type=None):
    '''
    update file with file_id, using contents from filepath

    assumptions:
        Infer mime type if not specified.
        filepath should be a valid file
        file_id must be specified
    '''
    assert os.path.isfile(filepath), f'{filepath} must be a valid file'
    assert file_id, 'must have valid file_id'

    fname = os.path.basename(filepath)
    file_ext = os.path.splitext(fname)[1]

    # lookup mime type
    if not mime_type:
        mime_type = mimetypes.types_map.get(file_ext, 'application/octet-stream')

    metadata = {}
    metadata['name'] = fname

    media = MediaFileUpload(filepath, mimetype=mime_type, resumable=True)

    file = service.files().update(body=metadata,
                                  media_body=media,
                                  fileId=file_id).execute()
    return file_id

def create_bytes(buf, filepath, parent=None, text=True):
    '''
    create filepath and fill content with buf.

    assumptions:
        write to root directory if parent not given
    '''
    if not parent: parent='root'

    fname = os.path.basename(filepath)
    #file_ext = os.path.splitext(fname)[1]

    # lookup mime type
    #mimetypes.init()
    #mime_type = mimetypes.types_map.get(file_ext, 'application/octet-stream')
    mime_type = 'text/plain' if text else 'application/octet-stream'

    metadata = {}
    metadata['name'] = fname
    metadata['parents'] = [parent]

    media = MediaInMemoryUpload(buf, mimetype=mime_type, resumable=True)

    file = service.files().create(body=metadata,
                                  media_body=media,
                                  fields='id').execute()
    print(f'File ID: {file["id"]}')
    return file['id']

def update_bytes(buf, filepath, file_id, text=True):
    '''
    update filepath content with buf

    assumptions:
        file_id must be specified
    '''
    assert file_id, 'must have valid file_id'

    fname = os.path.basename(filepath)
    #file_ext = os.path.splitext(fname)[1]

    # lookup mime type
    #mimetypes.init()
    #mime_type = mimetypes.types_map.get(file_ext, 'application/octet-stream')
    mime_type = 'text/plain' if text else 'application/octet-stream'

    metadata = {}
    metadata['name'] = fname

    media = MediaInMemoryUpload(buf, mimetype=mime_type, resumable=True)

    file = service.files().update(body=metadata,
                                  media_body=media,
                                  fileId=file_id).execute()

    return file_id

################################################################################
# user API functions
################################################################################
def save_file(src_filepath, dst_filepath=None, mimi_type=None):
    ''' upload file to google drive '''
    if not dst_filepath: dst_filepath = src_filepath
    assert not os.path.isabs(dst_filepath), 'support relative path only'
    assert os.path.isfile(src_filepath), f'{src_filepath} must be a valid file'
    # append root directory to dst_filepath
    dst_filepath = os.path.join(rootdir, dst_filepath)
    print(f'saving file to gdrive: {dst_filepath}')

    dst_basename = os.path.basename(dst_filepath)
    dst_dirname = os.path.dirname(dst_filepath)
    dst_dirname = os.path.normpath(dst_dirname)

    dirs = split_all(dst_dirname)
    print(dirs)
    folder = make_dirs(dirs)

    file_id = get_file(dst_basename, folder)
    if file_id:
        update_file(src_filepath, file_id)
    else:
        create_file(src_filepath, folder)

def save_to_file(dst_filepath, buf, text=True):
    ''' save buf to a remote file directly '''
    assert not os.path.isabs(dst_filepath), 'support relative path only'
    dst_filepath = os.path.join(rootdir, dst_filepath)
    print(f'saving buffer to gdrive: {dst_filepath}')

    dst_basename = os.path.basename(dst_filepath)
    dst_dirname = os.path.dirname(dst_filepath)
    dst_dirname = os.path.normpath(dst_dirname)

    dirs = split_all(dst_dirname)
    print(dirs)
    folder = make_dirs(dirs)

    if text: buf = buf.encode()

    file_id = get_file(dst_basename, folder)
    if file_id:
        update_bytes(buf, dst_basename, file_id, text=text)
    else:
        create_bytes(buf, dst_basename, folder, text=text)

def save_folder(src_filepath, dst_filepath=None, recursive=True):
    ''' recursively save of the contents in filepath '''
    if not dst_filepath: dst_filepath = src_filepath
    assert os.path.isdir(src_filepath), f'{src_filepath} must be a folder'
    assert not (dst_filepath.startswith('.') or dst_filepath.startswith('..')), \
           "dst_filepath cannot contain '.' or '..'"

    # recurse thru all files in src folder
    paths = glob.glob(f"{src_filepath}/**", recursive=True)
    for src_path in paths:
        dst_path = src_path.replace(src_filepath, dst_filepath, 1)
        print(f'copying {src_path} --> {dst_path}')

        if os.path.isdir(src_path):
            # skip, this will be taken care of in save_file()
            print('folder, skipping')
            pass
        elif os.path.isfile(src_path):
            save_file(src_path, dst_path)
        else:
            raise RuntimeError(f'cannot handle special files: {src_path}')


