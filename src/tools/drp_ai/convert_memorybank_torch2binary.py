import torch
import numpy as np
import argparse
from sys import byteorder
from struct import pack, unpack
from pathlib import Path

def convert(input_path, output_path):
    """convert parameter file.

    Args:
        input_path (str): input file path.
        output_path (str): output file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    w = torch.load(input_path, map_location='cpu')

    with open(output_path, 'wb') as f:
        f.write(b"PC")
        f.write(pack('f', float(w["thresould"])))
        f.write(pack('f', float(w["min_value"])))
        f.write(pack('f', float(w["max_value"])))
        f.write(pack('f', float(w["coreset_sampling_ratio"])))
        f.write(pack('i', int(w["num_neighbors"])))
    
        memory_bank = w["memory_bank"]
        f.write(pack('i', memory_bank.shape[0]))
        f.write(pack('i', memory_bank.shape[1]))

        memory_bank_bytes = memory_bank.detach().cpu().numpy().tobytes()
        f.write(memory_bank_bytes)

        #
        print("convert memorybank file...")
        print(f"  thresould: {float(w['thresould'])}")
        print(f"  min_value: {float(w['min_value'])}")
        print(f"  max_value: {float(w['max_value'])}")
        print(f"  coreset_sampling_ratio: {float(w['coreset_sampling_ratio'])}")
        print(f"  num_neighbors: {float(w['num_neighbors'])}")
        print(f"  memorybank size: ({memory_bank.shape[0]}, {memory_bank.shape[1]})")
        print()
        print(f"{input_path} -> {output_path}")
        print("done.")

def test_load(path):
    load_data = {}

    with open(path, 'rb') as f:
        b = f.read(2)
        if b != b'PC':
            raise ValueError("Invalid file format.") 
        
        b = f.read(4)
        v, = unpack('f', b)
        load_data["thresould"] = v

        b = f.read(4)
        v, = unpack('f', b)
        load_data["min_value"] = v

        b = f.read(4)
        v, = unpack('f', b)
        load_data["max_value"] = v
        
        b = f.read(4)
        v, = unpack('f', b)
        load_data["coreset_sampling_ratio"] = v

        b = f.read(4)
        v, = unpack('i', b)
        load_data["num_neighbors"] = v

        b = f.read(4)
        s1, = unpack('i', b)
        b = f.read(4)
        s2, = unpack('i', b)
        size = s1 * s2 * 4
        b = f.read(size)
        array = np.frombuffer(b, dtype=np.float32)
        load_data["memory_bank"] = array.reshape(s1, s2)
    return load_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='input path')
    parser.add_argument('output_path', help='output path')
    args = parser.parse_args()

    convert(args.input_path, args.output_path)
    
    #load_data = test_load(args.output_path)
    #print(load_data)
    #print(load_data["memory_bank"].shape)
