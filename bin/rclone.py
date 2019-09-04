#!/usr/bin/env python
#
# rclone copy tool (python executable)

REMOTE_NAME = 'gdrive'
ROOT_DIR = 'workspace'
RCLONE_CMD = 'rclone'
MAX_AGE = '5d'
DEBUG_EN = False

# copy options
# -v verbose
# -l preserve symlinks
# -u update mode, skip newer files
# --max-age 5d, only prcess files that are aged 5 days or newer
copy_cmd_list = [RCLONE_CMD, '-v', '-l', '-u', '--max-age', MAX_AGE, 'copy']
# sync options
# -P show progress
# -l preserve symlinks
# NOTE: consider doing a dry run with --dry-run flag as 
#       this operation is destructive to destination
sync_cmd_list = [RCLONE_CMD, '-P', '-l', 'sync']

if __name__ =="__main__":
    import sys 
    import os
    from pathlib import Path

    if len(sys.argv) != 3 or not sys.argv[1] in ('push','pull','mirror','publish'):
        sys.exit('usage: rclone.py (push|pull|mirror|publish) <file_path>')

    mode = sys.argv[1]
    fpath = sys.argv[2]
    abspath = os.path.abspath(fpath)
    #print(abspath)

    # special handling for file paths
    if os.path.isfile(abspath):
        dirpath = os.path.dirname(abspath)
        if mode in ('push','publish'):
            lpath, rpath = abspath, dirpath
        else:
            lpath, rpath = dirpath, abspath
    elif os.path.isdir(abspath):
        lpath = rpath = abspath
    else:
        raise RuntimeError('cannot handle special files')

    # construct relative path (for remote)
    idx = rpath.find(ROOT_DIR)
    prefix = rpath[:idx]
    relpath = os.path.relpath(rpath, prefix)
    #print(relpath)
    rpath = REMOTE_NAME + ':' + relpath

    if mode in ('push','publish'):
        srcpath, dstpath = lpath, rpath
    else:
        srcpath, dstpath = rpath, lpath

    # construct command list
    if mode in ('push','pull'):
        cmd_list = copy_cmd_list.copy()
    else:
        cmd_list = sync_cmd_list.copy()

    if DEBUG_EN:
        print('debug: performing dry run')
        cmd_list.append('--dry-run')

    cmd_list.extend([srcpath, dstpath])
    #print(cmd_list)

    print('updating dst path with src contents')
    cmd = ' '.join(cmd_list)
    print(cmd)

    # issue command
    import subprocess
    subprocess.run(cmd_list)



