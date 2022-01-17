import math
from time import sleep

import numpy as np

class DeviceMUmERAS():
    def __init__(self,**kwargs):

        ####General Simulation Parameters####
        self.tau = kwargs.get('tau',50e-3)
        self.V = kwargs.get('V',1e5)
        self.latency_step_size_array = kwargs.get('mhu_array',None)
        self.accuracy_step_size_array = kwargs.get('ni_array',None)
        self.accuracy_virtual_queue_init = kwargs.get('acc_virtual_queue_init',None)
        self.latency_virtual_queue_init = kwargs.get('latency_virtual_queue_init',None)
        self.device_number = kwargs.get('device_number',None)
        self.mean_arrivals_array = kwargs.get('mean_arrivals_array',None)
        self.channel_gain_stats = kwargs.get('channel_gain_vector_stats',None)

        self.path = kwargs.get('channel_vector_path',None)
        self.channel_gains = None

        ####Transmission parameters####
        self.band_array = kwargs.get('band_array',None)
        self.N0 = kwargs.get('N0',None)
        self.max_power = kwargs.get('max_power_array',None)

        ####Computation Parameters####
        self.device_frequency_matrix = kwargs.get('device_frequency_matrix',None)
        self.k_array = kwargs.get('k_array',None)

        ####Accuracy Parameters####
        self.accuracy_lut = kwargs.get('accuracy_lut',None)

        ####Other luts####
        self.M = kwargs.get('M_LUT',None)
        self.N = kwargs.get('N_LUT',None)
        self.J_d = kwargs.get('J_d',None)
        self.J_device_classifier = kwargs.get('J_device_classifier',None)
        self.compression_factor_lut = kwargs.get('compression_factor_lut',None)

        self.offloading_decision = kwargs.get('offloading_decision',np.array([1]))


    def simulate(self,Q_darray,Q_sarray,Z_array,Y_array,ts):

        best_frequency_allocation = np.zeros(self.device_number)
        best_rate_allocation = np.zeros(self.device_number)

        best_cf_allocation = np.zeros(self.device_number).astype(int)
        best_offloading_decision_allocation= np.zeros(self.device_number).astype(int)

        ####Device data####
        N_tx = np.zeros(self.device_number)
        new_arrivals = np.zeros(self.device_number)
        device_compression_energy = np.zeros(self.device_number)
        device_transmission_energy = np.zeros(self.device_number)
        accuracy = np.zeros(self.device_number)
        h = np.zeros(self.device_number)

        for device_index in range(self.device_number):
            cost_array = np.ones((self.compression_factor_lut[device_index].size,2))*math.inf
            best_frequency_array = np.zeros((self.compression_factor_lut[device_index].size,2))
            best_rate_array = np.zeros((self.compression_factor_lut[device_index].size,2))
            h[device_index] = self.compute_channel_gain(ts, device_index)**2
            for cf in self.compression_factor_lut[device_index]:
                for od in self.offloading_decision:
                    if cf>=6:
                        cf_ser = cf - 6
                    else:
                        cf_ser = cf
                    qtx = 14*self.latency_step_size_array[device_index]**2*(Q_darray[device_index]-od*Q_sarray[device_index][cf_ser])+self.latency_step_size_array[device_index]*Z_array[device_index]
                    if od==1:
                        if qtx>0:
                            min_cost = math.inf
                            best_rate = 0
                            best_freq = 0
                            for frequency in self.device_frequency_matrix[device_index]:
                                rate = self.compute_transmission_rate(qtx,Q_darray[device_index],h[device_index],cf,device_index,frequency)
                                cost = self.evaluate_cost_function(qtx,rate,h[device_index],device_index,frequency,cf,Y_array[device_index],od)
                                if cost<min_cost:
                                    min_cost = cost
                                    best_freq = frequency
                                    best_rate = rate
                            cost_array[cf,od]=min_cost
                            best_frequency_array[cf,od]=best_freq
                            best_rate_array[cf,od]=best_rate
                        else:
                            best_frequency_array[cf,od] = 0
                            cost_array[cf,od]=0
                    else:
                        min_cost = math.inf
                        best_freq = 0
                        rate = 0
                        for frequency in self.device_frequency_matrix[device_index]:
                            cost = self.evaluate_cost_function(qtx,rate,h[device_index],device_index,frequency,cf,Y_array[device_index],od)
                            if cost<min_cost:
                                min_cost = cost
                                best_freq = frequency
                        cost_array[cf,od]=min_cost
                        best_frequency_array[cf,od]=best_freq
                        best_rate_array[cf,od] = 0

            best_cf = np.unravel_index(cost_array.argmin(), cost_array.shape)[0]
            best_decision = np.unravel_index(cost_array.argmin(), cost_array.shape)[1]

            best_rate = best_rate_array[best_cf,best_decision]
            best_device_frequency = best_frequency_array[best_cf,best_decision]

            best_frequency_allocation[device_index] = best_device_frequency
            best_rate_allocation[device_index] = best_rate
            best_cf_allocation[device_index] = best_cf
            best_offloading_decision_allocation[device_index] =best_decision


        for device_index in range(self.device_number):
            if best_offloading_decision_allocation[device_index]==1:
                N_tx[device_index] = math.floor(self.tau*best_rate_allocation[device_index]/(self.M[device_index][best_cf_allocation[device_index]]*self.N[device_index][best_cf_allocation[device_index]]))
                '''if best_frequency_allocation[device_index]>0:
                    print(f'Best freq allocation: {best_frequency_allocation[device_index]}, ntx: {N_tx[device_index]}, ts: {ts}, device: {device_index}')'''
            else:
                N_tx[device_index] = math.floor(self.tau*best_frequency_allocation[device_index]*self.J_device_classifier[device_index][best_cf_allocation[device_index]])
            new_arrivals[device_index] = np.random.poisson(self.mean_arrivals_array[device_index])
            device_compression_energy[device_index] = self.get_compression_energy_term(device_index,best_frequency_allocation[device_index])
            device_transmission_energy[device_index] = self.get_transmission_energy_term(device_index,h[device_index],best_rate_allocation[device_index])
            if best_rate_allocation[device_index]==0 and best_offloading_decision_allocation[device_index]==1:
                accuracy[device_index] = 0
            else:
                accuracy[device_index] = self.accuracy_lut[device_index][best_cf_allocation[device_index]]

        return N_tx,new_arrivals,accuracy,best_cf_allocation,device_compression_energy,device_transmission_energy,best_rate_allocation,best_frequency_allocation,best_offloading_decision_allocation

    ####Cost function evaluation####
    def evaluate_cost_function(self,qtx,rate,h,device_index,frequency,compression_factor,yk,offloading_decision):
        return self.get_latency_term(qtx,rate,device_index,compression_factor,offloading_decision,frequency)+\
               self.V*self.get_transmission_energy_term(device_index,h,rate)+\
               self.V*self.get_compression_energy_term(device_index,frequency)+\
               self.get_accuracy_term(device_index,compression_factor,yk)

    ####Energy terms of the cost function####
    def get_transmission_energy_term(self,device_index,h,rate):
        return (self.tau*self.band_array[device_index]*self.N0/h)*(math.exp((rate*math.log(2))/self.band_array[device_index])-1)

    def get_compression_energy_term(self,device_index,frequency):
        return self.k_array[device_index]*self.tau*frequency**3

    ####Latency and accuracy terms of the cost function####
    def get_latency_term(self,qtx,rate,device_index,compression_factor,offloading_decision,frequency):
        return -qtx*(offloading_decision*(self.tau*rate)/(self.M[device_index][compression_factor]*self.N[device_index][compression_factor])+(1-offloading_decision)*self.J_device_classifier[device_index][compression_factor]*frequency*self.tau)

    def get_accuracy_term(self,device_index,compression_factor,yk):
        return -self.accuracy_step_size_array[device_index]*yk*self.accuracy_lut[device_index][compression_factor]

    ####Computing transmission energy####
    def compute_transmission_rate(self,qtx,qkd,h,compression_factor,device_index,frequency):

        transmission_rate = (self.band_array[device_index]/math.log(2))*math.log((qtx*h)/(self.V*self.N0*self.M[device_index][compression_factor]*self.N[device_index][compression_factor]))
        transmission_rate = max(0,transmission_rate)

        max_rate_shannon = self.band_array[device_index]*math.log2(1+(self.max_power[device_index]*h)/(self.band_array[device_index]*self.N0))

        max_rate = min(max_rate_shannon,
                       (((qkd+ 1) * (self.M[device_index][compression_factor] * self.N[device_index][compression_factor])))/ self.tau,
                       self.J_d[device_index][compression_factor] * frequency * self.M[device_index][compression_factor] * self.N[device_index][compression_factor])

        u_bound = min(transmission_rate, max_rate)
        return max(0, u_bound)

    def compute_channel_gain(self,ts,device_index):
        self.channel_gains = np.load(self.path[device_index],allow_pickle=True).ravel()
        return self.channel_gains[ts]


