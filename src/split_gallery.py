import glob
import shutil
import os
import random

src_dir = "./images5o_crop/"
src_dir2 = "./images5ox1_crop/"

dst_dir = "./images5o_query/"
dst_dir2 = "./images5o_gallery/"

dst_dir3 = "./images5ox1_query/"
dst_dir4 = "./images5ox1_gallery/"

all = glob.iglob(os.path.join(src_dir, "*.jpg"))
all2 = glob.iglob(os.path.join(src_dir2, "*.jpg"))
# print(enumerate(all))
# n = 0
# arr = os.listdir(src_dir)
# print(arr)
# l = set(arr)
l = set()
while True:
    if len(l) == 3398:
        break
    else:
        l.add(random.randint(0,6795))
# print(l)

n = 0
m = 0
for i, j in enumerate(all):
    # print(type(j[-16:]), type(l))
    if i in l:
        n = n+1
        shutil.copy(j, dst_dir2)
    else:
        m = m +1
        shutil.copy(j, dst_dir)
    # print(i,j)
    # n = n+1

print('1st',n,m)

n = 0
m = 0
for i, j in enumerate(all2):
    # print(type(j[-16:]), type(l))
    if i in l:
        n = n + 1
        shutil.copy(j, dst_dir4)
    else:
        m = m + 1
        shutil.copy(j, dst_dir3)
    # print(i,j)
    # n = n+1

print('2nd',n, m)


# for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
#     shutil.copy(jpgfile, dst_dir)