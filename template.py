import os

dirs_ = [
    os.path.join("data","raw"),
    os.path.join("data","processed"),
    "notebooks",
    "saved_models",
    os.path.join("src","components"),
    os.path.join("src","pipeline"),
    "logs"
]

for dir_ in dirs_:
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_,".gitkeep"),"w") as f:
        pass

files_ = [
    "dvc.yaml",
    "params.yaml",
    ".gitignore",
    os.path.join("src","__init__.py"),
    os.path.join("src","components","__init__.py"),
    os.path.join("src","pipeline"),"__init__.py",
    "README.md"
]

for file_ in files_:
    with open(file_, "a") as f:
        pass