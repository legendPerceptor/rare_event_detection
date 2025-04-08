import argparse
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def get_one_layer_and_dark_diff(dimension: list[int], data_file: str, dark_file:str, layer_number: int, data_type: np.dtype, skip_size=0):
    if len(dimension) != 3: 
        # error, this function only deals with 3D tensor
        return (-1, -1, -1)
    
    layer_size = dimension[0] * dimension[1]

    data_type = np.dtype(data_type) # make sure the type is actually a np.dtype object

    with open(data_file, 'rb') as f:
        f.seek(skip_size + layer_number * layer_size * data_type.itemsize)
        layer_data = np.fromfile(f, dtype=data_type, count=layer_size)
        layer_data = layer_data.reshape(dimension[0], dimension[1])

    with open(dark_file, 'rb') as f:
        dark_data = np.fromfile(f, dtype=data_type, count=layer_size)
        dark_data = dark_data.reshape(dimension[0], dimension[1])

    diff = layer_data - dark_data

    print("diff: ", diff[:10, :10])
    print("dark: ", dark_data[:10, :10])
    print("layer: ", layer_data[:10, :10])

    data_min, data_max = np.min(layer_data), 2000 # np.max(layer_data)
    diff_min, diff_max = np.min(diff), np.max(diff) # np.max(layer_data)

    return (layer_data, data_min, data_max), (diff, diff_min, diff_max)



def dtype_arg(value: str) -> np.dtype:
    """
    Converts a string to a valid NumPy dtype.
    Raises ArgumentTypeError if the string is not recognized.
    """
    try:
        return np.dtype(value)
    except TypeError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid NumPy dtype")
    

def main():
    parser = argparse.ArgumentParser(description="Run the rare event detection pipeline.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_file", type=str, default="test.edf.ge5",
                        help="Path to the data file.")
    
    parser.add_argument("--decompressed_data_file", type=str, default="test.edf.ge5.sz3split.dp",
                        help="Path to the decompressed data file")
    
    parser.add_argument("--dark_file", type=str, default="dark_4_test.edf.ge5",
                        help="Path to the dark file.")
    
    parser.add_argument("--dimension", type=int, nargs='+', default=[2048, 2048, 1440],
                        help="The dimension of the data.")
    
    parser.add_argument("--layer", type=int, default=0,
                        help="Select which layer to visualize.")
    
    parser.add_argument("--data_type", type=dtype_arg, default=np.uint16,
                        help="datatype: uint16, float32, float64, etc. Default is uint16.")
    
    parser.add_argument("--save_fig_path", type=str, default=None,
                        help="Path to save the figure.")
    
    parser.add_argument("--skip_size", type=int, default=8396800, help="Select the header size to skip.")
    
    args = parser.parse_args()
    (raw_data, raw_data_min, raw_data_max), (diff_raw_data, diff_raw_min, diff_raw_max)  = get_one_layer_and_dark_diff(args.dimension, args.data_file, args.dark_file, args.layer, args.data_type, args.skip_size)

    print("finished getting the layer data of the original file.")

    (dp_image_data, dp_data_min, dp_data_max), (diff_dp_image_data, diff_dp_data_min, diff_dp_data_max) = get_one_layer_and_dark_diff(args.dimension, args.decompressed_data_file, args.dark_file, args.layer, args.data_type, args.skip_size)

    print("finished getting the preview data of the decompressed file.")

    dimension = args.dimension

    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    ax = axes[0][0]
    ax.imshow(raw_data, cmap=plt.get_cmap('rainbow'), extent=[0, dimension[0], 0, dimension[1]], aspect="equal",
              norm=plt.Normalize(vmin=raw_data_min, vmax=raw_data_max))
    ax.set_title(f"Original image at layer {args.layer}")
    ax.set_xlim(0, args.dimension[0])
    ax.set_ylim(0, args.dimension[1])

    ax = axes[0][1]
    ax.imshow(diff_raw_data, cmap=plt.get_cmap('rainbow'), extent=[0, dimension[0], 0, dimension[1]], aspect="equal",
              norm=plt.Normalize(vmin=diff_raw_min, vmax=diff_raw_max))
    ax.set_title(f"Original diff at layer {args.layer}")
    ax.set_xlim(0, args.dimension[0])
    ax.set_ylim(0, args.dimension[1])

    ax = axes[1][0]
    ax.imshow(dp_image_data, cmap=plt.get_cmap('rainbow'), extent=[0, dimension[0], 0, dimension[1]], aspect="equal",
              norm=plt.Normalize(vmin=dp_data_min, vmax=dp_data_max))
    ax.set_title(f"Decompressed image at layer {args.layer}")
    ax.set_xlim(0, args.dimension[0])
    ax.set_ylim(0, args.dimension[1])

    ax = axes[1][1]
    ax.imshow(diff_dp_image_data, cmap=plt.get_cmap('rainbow'), extent=[0, dimension[0], 0, dimension[1]], aspect="equal",
              norm=plt.Normalize(vmin=diff_dp_data_min, vmax=diff_dp_data_max))
    ax.set_title(f"Decompressed diff at layer {args.layer}")
    ax.set_xlim(0, args.dimension[0])
    ax.set_ylim(0, args.dimension[1])

    figure.suptitle("Integer-SZ3, eb=200, CR=24")

    if args.save_fig_path is not None:
        plt.tight_layout()
        plt.savefig(args.save_fig_path, bbox_inches='tight', dpi=300)
        print(f"figure saved to {args.save_fig_path}")
    else:
        plt.show()

    print("The visualization program has finished.")
    


if __name__ == "__main__":
    main()


