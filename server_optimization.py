import math
import queue

import numpy as np


class ServerMUmERAS():
    def __init__(self,**kwargs):

        ####General simulation parameters####
        self.k_ser = kwargs.get('k_ser',None)
        self.V = kwargs.get('V',None)
        self.tau = kwargs.get('tau',None)

        self.server_frequencies = kwargs.get('server_frequencies_array',None)
        self.device_number = kwargs.get('device_number',None)

        self.latency_step_size_array = kwargs.get('mhu_array',None)
        self.J_ser = kwargs.get('j_ser',None)

        self.compression_factor_array = kwargs.get('compression_factor_array',None)


    def simulate(self,Q_ser,Z):
        frequency_matrix = np.zeros((self.server_frequencies.size,self.device_number,int(self.compression_factor_array[0].size/2)))
        n_task_matrix = np.zeros((self.server_frequencies.size,self.device_number,int(self.compression_factor_array[0].size/2)))
        Q_comp = np.zeros((self.device_number,int(self.compression_factor_array[0].size/2)))
        for dev in range(self.device_number):
            Q_comp[dev] = 14*self.latency_step_size_array[dev]**2*Q_ser[dev]+self.latency_step_size_array[dev]*Z[dev]*np.ones(int(self.compression_factor_array[0].size/2))
        frequency_index = 0
        for current_frequency in self.server_frequencies:
            check_matrix = np.ones((self.device_number,self.compression_factor_array[0].size))
            residual_frequency = current_frequency
            while(residual_frequency>0):
                max=-math.inf
                dev_max=None
                for dev in range(self.device_number):
                    for cf in range(int(self.compression_factor_array[0].size/2)):
                        if check_matrix[dev][cf]==1:
                            weight = Q_comp[dev][cf]*self.J_ser[cf]
                            if(weight>=max):
                                max = weight
                                dev_max = dev
                                best_cf = cf

                if dev_max==None:
                    break

                F_dev_cf = min((Q_ser[dev_max][best_cf])/(self.J_ser[best_cf]*self.tau),residual_frequency)
                #print(f'{Q_ser[dev_max][best_cf]}')
                frequency_matrix[frequency_index][dev_max][best_cf] = F_dev_cf
                residual_frequency = residual_frequency - F_dev_cf
                check_matrix[dev_max][best_cf] = 0

            frequency_index+=1
        i=0
        best_cost= math.inf
        best_i = 0
        for frequency in self.server_frequencies:
            cost = self.evaluate_cost_function(Q_comp,frequency_matrix[i],frequency)
            #print(f'Frequenza: {frequency/1e9}, costo: {cost}')
            if cost<best_cost:
                best_i = i
                best_cost = cost
            i+=1
        energy = self.compute_energy_term(self.server_frequencies[best_i])

        for n_d in range(self.device_number):
            for cf in range(int(self.compression_factor_array[0].size/2)):
                n_task_matrix[best_i][n_d][cf] = math.floor(frequency_matrix[best_i][n_d][cf]*self.J_ser[cf]*self.tau)
        #print(f'Ho scelto la frequenza: {self.server_frequencies[best_i]/1e9}GHz')
        return n_task_matrix[best_i],frequency_matrix[best_i],energy,self.server_frequencies[best_i]

    ####Cost function computing####
    def evaluate_cost_function(self,Q_comp,frequency_division,frequency):
        return self.tau*self.compute_latency_term(Q_comp,frequency_division)+(self.V)*self.compute_energy_term(frequency)

    def compute_latency_term(self,Q_comp,frequency_division):
        lat = 0
        for dev in range(self.device_number):
            for cf in range(int(self.compression_factor_array[0].size/2)):
                lat+=Q_comp[dev][cf]*frequency_division[dev][cf]*self.J_ser[cf]
        return -lat


    def compute_energy_term(self,frequency):
        return self.tau*self.k_ser*frequency**3

