import argparse
import pickle
import config

from vis_utils import parse_el, parse_th, parse_mf, plot_animation

def main():
    parser = argparse.ArgumentParser(
        description="Load a datapoint from a pickle file and process its content."
    )
    parser.add_argument(
        "--dp-path",
        type=str,
        default="/Users/gapaza/repos/ideal/zaratan-deployment/datasets/thermoelastic2dv8/vdxckdzaxyun.pkl",
        help="Path to the datapoint pickle file."
    )
    args = parser.parse_args()

    # Load the datapoint
    with open(args.dp_path, 'rb') as f:
        datapoint = pickle.load(f)

    # Print keys of datapoint
    print(datapoint.keys())

    # --- SAVE DIR ---
    save_dir = config.plots_dir

    # --- ELASTIC ---
    elastic_data = datapoint['elastic']
    parse_el(elastic_data, save_dir)
    # plot_animation(elastic_data, title='design_history_elastic', save_dir=save_dir)

    # # --- THERMAL ---
    # thermal_data = datapoint['thermal']
    # parse_th(thermal_data, save_dir)
    # # plot_animation(thermal_data, title='design_history_thermal', save_dir=save_dir)

    # # --- THERMOELASTIC ---
    # thermoelastic_data = datapoint['thermoelastic']
    # parse_mf(thermoelastic_data, save_dir)
    # # plot_animation(thermoelastic_data, title='design_history_thermoelastic', save_dir=save_dir)

    

    # # --- MNM (Optional) ---
    # mnm_data = datapoint['mnm_th_to_el_0']
    # parse_el(mnm_data, save_dir, sup_title='mnm_th_to_el')
    # plot_animation(mnm_data, title='design_history_mnm_th_to_el', save_dir=save_dir)

    # mnm_data = datapoint['mnm_el_to_th_0']
    # parse_th(mnm_data, save_dir, sup_title='mnm_el_to_th')
    # plot_animation(mnm_data, title='design_history_mnm_el_to_th', save_dir=save_dir)

    # # --- VFM (Optional) ---
    # vfm_data = datapoint['vfm_el_el_0']
    # parse_el(vfm_data, save_dir, sup_title='vfm_el_el_0')
    # plot_animation(vfm_data, title='design_history_vfm_el_el_0', save_dir=save_dir)

    # vfm_data = datapoint['vfm_el_el_1']
    # parse_el(vfm_data, save_dir, sup_title='vfm_el_el_1')
    # plot_animation(vfm_data, title='design_history_vfm_el_el_1', save_dir=save_dir)

    # vfm_data = datapoint['vfm_th_th_0']
    # parse_th(vfm_data, save_dir, sup_title='vfm_th_th_0')
    # plot_animation(vfm_data, title='design_history_vfm_th_th_0', save_dir=save_dir)

    # vfm_data = datapoint['vfm_th_th_1']
    # parse_th(vfm_data, save_dir, sup_title='vfm_th_th_1')
    # plot_animation(vfm_data, title='design_history_vfm_th_th_1', save_dir=save_dir)


if __name__ == '__main__':
    main()