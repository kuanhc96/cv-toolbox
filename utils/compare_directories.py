import os

# Check to see if the directory for predict has the same structure of
# the directory that was provided for training
def _isEqualSubDirs(dir1, dir2):
    dir1SubDirs = next(os.walk(dir1))[1]
    dir2SubDirs = next(os.walk(dir2))[1]

    if len(dir1SubDirs) != len(dir2SubDirs):
        return False

    dir1SubDirs.sort()
    dir2SubDirs.sort()

    for (subDir1, subDir2) in zip(dir1SubDirs, dir2SubDirs):
        if subDir1 != subDir2:
            return False
    return True