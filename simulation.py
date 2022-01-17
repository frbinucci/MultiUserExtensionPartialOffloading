class Simulation():

    def __init__(self,**kwargs):

        ####Queues Status####
        self.latency_virtual_queue = kwargs.get('latency_virtual_queue',None)
        self.accuracy_virtual_queue = kwargs.get('accuracy_virtual_queue',None)
        self.qtot = kwargs.get('q_tot',None)

        ####Energy Status####
        self.compression_energy_array = kwargs.get('compression_energy_array',None)
        self.transmission_energy_array = kwargs.get('transmission_energy_array',None)
        self.server_energy_array = kwargs.get('server_energy_array',None)
        self.energy_array = kwargs.get('energy_array',None)

        ####Accuracy####
        self.accuracy_status = kwargs.get('accuracy_status',None)

        ##Speed parameters##
        self.rate_array = kwargs.get('rate_array',None)
        self.device_frequencies_array = kwargs.get('device_freq_array',None)
        self.server_frequencies_array = kwargs.get('server_freq_array',None)

        self.offloading_decision = kwargs.get('offloading_decisions',None)
