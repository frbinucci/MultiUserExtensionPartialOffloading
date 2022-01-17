import math
import os
import pickle
import sys
from datetime import datetime
from decimal import Decimal

import numpy as np

import simulation
from device_optimization import DeviceMUmERAS
from server_optimization import ServerMUmERAS


class OptimizationStrategy():

    def __init__(self,**kwargs):

        self.tau = kwargs.get('tau',None)
        self.V = kwargs.get('V',None)
        self.latency_step_size_array = kwargs.get('mhu_array',None)
        self.accuracy_step_size_array = kwargs.get('ni_array',None)
        self.accuracy_virtual_queue_init = kwargs.get('accuracy_virtual_queue_init',None)
        self.latency_virtual_queue_init = kwargs.get('latency_virtual_queue_init',None)
        self.mean_arrivals_array = kwargs.get('mean_arrivals_array', None)
        self.device_number = kwargs.get('device_number',10)
        self.band_array = kwargs.get('band_array',None)

        self.J_server = kwargs.get('J_s',None)

        self.N0=kwargs.get('N0',None)

        ####Constraints definition####
        self.latency_constraints_array = kwargs.get('latency_constraints_array',None)
        self.accuracy_constraints_array = kwargs.get('accuracy_constraints_array',None)

        self.max_power_array = kwargs.get('max_power_array',None)
        self.J_d = kwargs.get('J_d',None)
        self.J_device_classifier = kwargs.get('J_device_classifier',None)
        self.N_lut = kwargs.get('N_lut',None)
        self.G_lut = kwargs.get('G_lut',None)
        self.M_lut = kwargs.get('M_lut',None)
        self.Cf_lut = kwargs.get('Cf_lut',None)

        self.device_frequency_matrix = kwargs.get('device_frequency_matrix',None)
        self.k_array = kwargs.get('k_array',None)

        self.k_ser = kwargs.get('k_ser',None)
        self.server_frequencies_array = kwargs.get('server_frequencies_array',None)

        self.time_slot_number=kwargs.get('simulation_duration',10000)
        self.channel_gain_stats = kwargs.get('channel_gain_stats',None)

        self.avg_compression_factor_window_length = kwargs.get('avg_compression_factor_window_length',10)
        self.window_positions = np.zeros(self.device_number).astype(int)

        self.output_dir = kwargs.get('output_dir',None)

        self.channel_gain_path = kwargs.get('channel_gain_path',None)

    def run_optimization_strategy(self):
        print("Simulation started at: "+datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
        device_optimizer = DeviceMUmERAS(tau=self.tau,
                                         V=self.V,
                                         mhu_array=self.latency_step_size_array,
                                         ni_array=self.accuracy_step_size_array,
                                         acc_virtual_queue_init=self.accuracy_virtual_queue_init,
                                         latency_virtual_queue_init=self.latency_virtual_queue_init,
                                         device_number=self.device_number,
                                         mean_arrivals_array=self.mean_arrivals_array,
                                         channel_gain_vector_stats=self.channel_gain_stats,
                                         band_array=self.band_array,
                                         N0=self.N0,
                                         max_power_array=self.max_power_array,
                                         device_frequency_matrix=self.device_frequency_matrix,
                                         k_array=self.k_array,
                                         accuracy_lut=self.G_lut,
                                         M_LUT=self.M_lut,
                                         N_LUT=self.N_lut,
                                         J_d=self.J_d,
                                         J_device_classifier = self.J_device_classifier,
                                         compression_factor_lut=self.Cf_lut,
                                         channel_vector_path=self.channel_gain_path,
                                         offloading_decision=np.array([0,1]))

        server_optimizer = ServerMUmERAS(k_ser=self.k_ser,
                                         tau=self.tau,
                                         V=self.V,
                                         server_frequencies_array=self.server_frequencies_array,
                                         j_ser=self.J_server,
                                         device_number=self.device_number,
                                         mhu_array=self.latency_step_size_array,
                                         compression_factor_array=self.Cf_lut)

        #INIT QUEUES
        Q_dev = np.random.poisson(self.mean_arrivals_array)
        Q_ser = np.zeros((self.device_number,6))

        Z = self.latency_virtual_queue_init
        Y = self.accuracy_virtual_queue_init


        ####Creating list arrays####
        Z_array = np.zeros((self.device_number,self.time_slot_number))
        Y_array = np.zeros((self.device_number,self.time_slot_number))
        accuracy_status = np.zeros((self.device_number,self.time_slot_number))
        Q_tot_array = np.zeros((self.device_number,self.time_slot_number))

        rate_array = np.zeros((self.device_number,self.time_slot_number))
        freq_array = np.zeros((self.device_number,self.time_slot_number))
        server_freq_array = np.zeros(self.time_slot_number)

        compression_energy_array = np.zeros((self.device_number,self.time_slot_number))
        offloading_decision_status = np.zeros((self.device_number,self.time_slot_number))
        transmission_energy_array = np.zeros((self.device_number,self.time_slot_number))
        server_energy_array = np.zeros((self.time_slot_number))

        percent_status = 0

        for ts in range(self.time_slot_number):
            percent_status+=1
            percentage = math.floor((ts/self.time_slot_number)*100)
            sys.stdout.write('Status: %s %%\r' % percentage)
            N_tx,new_arrivals,accuracy,compression_factor,device_compression_energy,device_transmission_energy,best_rate,best_frequency,best_offloading_decision = device_optimizer.simulate(Q_dev,Q_ser,Z,Y,ts)
            N_ser,frequency_allocation,server_energy,server_freq = server_optimizer.simulate(Q_ser,Z)

            for k in range(self.device_number):
                for cf in range(int(self.Cf_lut[0].size/2)):
                    Q_ser[k][cf]=max(0,Q_ser[k][cf]-N_ser[k][cf])

            for dev in range(self.device_number):
                if compression_factor[dev]<6:
                    cf = compression_factor[dev]
                else:
                    cf = compression_factor[dev]-6
                if best_offloading_decision[dev]==1:
                    Q_ser[dev][cf]+=min(N_tx[dev],Q_dev[dev])

            #Queue updates
            Q_dev = np.maximum(np.zeros(self.device_number), Q_dev - N_tx) + new_arrivals

            Q_tot = np.copy(Q_dev)

            for cf in range(int(self.Cf_lut[0].size/2)):
                Q_tot+=Q_ser[:,cf]

            Z = np.maximum(0,Z+self.latency_step_size_array*(Q_tot-self.latency_constraints_array))
            for i in range(Y.size):
                if accuracy[i]!=0:
                    Y[i] = max(0,Y[i]+self.accuracy_step_size_array[i]*(self.accuracy_constraints_array[i]-accuracy[i]))
            #Y = np.maximum(0,Y+self.accuracy_step_size_array*(self.accuracy_constraints_array-accuracy))

            ####Queue tracking####
            Q_tot_array[:,ts]=Q_tot
            Z_array[:, ts]=Z
            Y_array[:, ts]=Y
            rate_array[:,ts]= best_rate
            freq_array[:,ts]= best_frequency
            server_freq_array[ts] = server_freq
            offloading_decision_status[:,ts] = best_offloading_decision

            ####Energy tracking###
            compression_energy_array[:,ts]=device_compression_energy
            transmission_energy_array[:,ts]=device_transmission_energy
            server_energy_array[ts]=server_energy
            accuracy_status[:,ts] = accuracy

        simulation_data = simulation.Simulation(latency_virtual_queue=Z_array,
                                                accuracy_virtual_queue=Y_array,
                                                q_tot=Q_tot_array,
                                                transmission_energy_array=transmission_energy_array,
                                                compression_energy_array=compression_energy_array,
                                                server_energy_array=server_energy_array,
                                                accuracy_status=accuracy_status,
                                                server_freq_array=server_freq_array,
                                                rate_array=rate_array,
                                                device_freq_array=freq_array,
                                                offloading_decisions=offloading_decision_status)
        self.save_object(simulation_data)
        print("Simulation completed at: "+datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))


    def save_object(self,simulation):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        filename =  'simulation_'+self.format_e(Decimal(self.V))
        path = self.output_dir+'/'+filename+'.dat'
        with open(path, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(simulation, outp, pickle.HIGHEST_PROTOCOL)

    def format_e(self,n):
        a = '%E' % n
        return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]






