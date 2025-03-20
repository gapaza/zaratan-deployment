import argparse
import pickle

from vis_utils import parse_el, parse_th, parse_mf, plot_animation

def main():
    parser = argparse.ArgumentParser(
        description="Load a datapoint from a pickle file and process its content."
    )
    parser.add_argument(
        "--dp-path",
        type=str,
        default="/Users/gapaza/repos/datasets/thermoelastic2d/atbsuohzgyqt.pkl",
        help="Path to the datapoint pickle file."
    )
    args = parser.parse_args()

    # Load the datapoint
    with open(args.dp_path, 'rb') as f:
        datapoint = pickle.load(f)

    # Print keys of datapoint
    print(datapoint.keys())

    # --- ELASTIC ---
    elastic_data = datapoint['elastic']
    parse_el(elastic_data)
    plot_animation(elastic_data, title='design_history_elastic')

    # --- THERMAL ---
    thermal_data = datapoint['thermal']
    parse_th(thermal_data)
    plot_animation(thermal_data, title='design_history_thermal')

    # --- THERMOELASTIC ---
    thermoelastic_data = datapoint['thermoelastic']
    parse_mf(thermoelastic_data)
    plot_animation(thermoelastic_data, title='design_history_thermoelastic')

if __name__ == '__main__':
    main()