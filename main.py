# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse

import numpy as np

from optimization_strategy import OptimizationStrategy


def print_hi(name):
    device_number = 5
    ####Devices constraints####
    latency_constraints_array=np.array([3,3,3,3,3])
    accuracy_constraints_array = np.array([0.95,0.95,0.95,0.95,0.95])
    mean_arrivals_rate_array = np.array([1,1,1,1,1])

    ####Virtual Queues####
    latency_step_size_array = 7*np.array([1,1,1,1,1])
    accuracy_step_size_array = 500*np.array([1,1,1,1,1])
    accuracy_virtual_queue_init = 0*np.array([0,0,0,0,0]).astype(float)
    latency_virtual_queue_init = 0*np.array([1,1,1,1,1])

    ####Device computational features####
    number_frequencies=10
    step=0.1
    base_frequency=1.4e9
    k_array = np.array([1.097e-27,1.097e-27,1.097e-27,1.097e-27,1.097e-27])
    device_frequency_array=np.tile(np.arange(1,number_frequencies+1)*step*base_frequency,(device_number,1))

    ####Transmission Features####
    N0=1.25e-20
    band_array = np.array([2.5e6,2.5e6,2.5e6,2.5e6,2.5e6])
    max_power_array = np.array([1e4,1e4,1e4,1e4,1e4])

    ####LUTs####
    accuracy_lut = np.array([0.973,
                             0.965,
                             0.93,
                             0.918,
                             0.83,
                             0.67,
                             0.973,
                             0.955,
                             0.915,
                             0.90,
                             0.77,
                             0.50])

    jd_lut =np.array([6.67e-8,
                      8.60e-8,
                      8.98e-8,
                      9.47e-8,
                      1.35e-7,
                      1.32e-7,
                      6.67e-8,
                      1.04e-7,
                      1.27e-7,
                      1.57e-7,
                      2.25e-7,
                      2.25e-7
                      ])

    N_lut = np.array([1.08,
                      2.27,
                      4.72,
                      9.06,
                      8,
                      8,
                      1.08,
                      2.27,
                      4.72,
                      9.06,
                      8,
                      8
                      ])

    M_lut = np.array([49152,
                      12288,
                      3072,
                      768,
                      192,
                      48,
                      49152,
                      12288,
                      3072,
                      768,
                      192,
                      48
                      ])

    J_s = np.array([1.2e-7,
                      2.17e-7,
                      2.87e-7,
                      3.57e-7,
                      5e-7,
                      6.25e-7
                      ])

    J_device_classifier = np.array([4.44e-8,
                      6.16e-8,
                      6.84e-8,
                      7.48e-8,
                      1.06e-7,
                      1.08e-7,
                      4.44e-8,
                      7.03e-8,
                      8.8e-8,
                      1.09e-7,
                      1.55e-7,
                      1.65e-7
                      ])

    accuracy_matrix = np.tile(accuracy_lut,(device_number,1))
    jd_matrix = np.tile(jd_lut,(device_number,1))
    J_device_classifier = np.tile(J_device_classifier,(device_number,1))
    N_matrix = np.tile(N_lut,(device_number,1))
    M_matrix = np.tile(M_lut,(device_number,1))

    compression_factor_matrix = np.tile(np.array([0,1,2,3,4,5,6,7,8,9,10,11]),(device_number,1)).astype(int)

    ####Server parameters####
    k_ser = 1.097e-27
    number_frequencies=10
    step=0.1
    base_frequency=4.5e9
    server_frequency_array=np.arange(0,number_frequencies+1)*step*base_frequency

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--out', help='Output directory',default='./')
    parser.add_argument('--tau',help='Time slot duration',default=50e-3,type=float)
    parser.add_argument('--V',help='Trade-off parameter (V)',default=1e3,type=float)
    parser.add_argument('--n_slots',help='Simulation duration',default=int(2e3),type=int)

    args = parser.parse_args()

    output_dir = args.out
    tau = args.tau
    V = args.V
    sim_duration = args.n_slots

    MERAS = OptimizationStrategy(tau=tau,
                                 V=V,
                                 device_number=device_number,
                                 latency_virtual_queue_init=latency_virtual_queue_init,
                                 accuracy_virtual_queue_init=accuracy_virtual_queue_init,
                                 simulation_duration=sim_duration,
                                 N0=N0,
                                 band_array=band_array,
                                 mhu_array=latency_step_size_array,
                                 ni_array=accuracy_step_size_array,
                                 J_s=J_s,
                                 G_lut=accuracy_matrix,
                                 J_d=jd_matrix,
                                 N_lut=N_matrix,
                                 M_lut=M_matrix,
                                 Cf_lut=compression_factor_matrix,
                                 device_frequency_matrix=device_frequency_array,
                                 k_array=k_array,
                                 k_ser=k_ser,
                                 server_frequencies_array=server_frequency_array,
                                 max_power_array=max_power_array,
                                 latency_constraints_array=latency_constraints_array,
                                 accuracy_constraints_array=accuracy_constraints_array,
                                 mean_arrivals_array=mean_arrivals_rate_array,
                                 channel_gain_path=['test.npy','test.npy','test.npy','test.npy','test.npy'],
                                 output_dir = output_dir,
                                 J_device_classifier=J_device_classifier
                                 )

    MERAS.run_optimization_strategy()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
