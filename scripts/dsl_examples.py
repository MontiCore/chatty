import os
import shutil
import subprocess

def scrape_examples(project, endings, dest):
    if not os.path.exists(dest):
        os.makedirs(dest)
    invalid_files = []
    for root, dirs, files in os.walk(project):
        for file in files:
            for ending in endings:
                if file.endswith(ending):
                    source_file = os.path.join(root, file)
                    if ending == ".umlp":
                        destination_file = os.path.join(dest, file.split(".")[0] + ".cd")
                    else:
                        destination_file = os.path.join(dest, file)
                    if validate_examples(source_file):
                        shutil.copyfile(source_file, destination_file)
                        print(f"Copied {file} to {destination_file}")
                    else:
                        invalid_files.append(file)
                        print(f"File {file} did not pass syntax check")
    print(f"Number of invalid files: {len(invalid_files)}\n"
          f"Invalid files: {invalid_files}")
    with open("invalid.txt", "w") as f:
        f.write("\n".join(invalid_files))

def validate_examples(file_path):
    with open(file_path, "rb") as f:
        diagram = f.read()
    command = ["java", "-jar", "../MCCD.jar", "--stdin"]
    try:
        subprocess.run(command, input=diagram, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        return False

if __name__ == "__main__":
    project_path = "../../cd4analysis"
    file_endings = [".umlp", ".cd"]
    dest = "../data/cd4a"
    scrape_examples(project_path, file_endings, dest)

