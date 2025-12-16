import json
import os.path


def generate_c_files(json_file,output_folder):
    with open(json_file,'r') as f:
        data = json.load(f)
    num_digits = len(str(len(data)))
    for idx,obj in enumerate(data):
        func = obj['func']
        target = obj['target']
        filename = '{:0{width}}_{}.c'.format(idx,target,width=num_digits)
        print(filename)
        filepath = os.path.join(output_folder,filename)
        print(filepath)
        with open(filepath,"w") as c_files:
            c_files.write(func)
json_file = 'function.json'
output_folder = 'c_file'
generate_c_files(json_file,output_folder)
files =os.listdir(output_folder)
print(len((files)))


