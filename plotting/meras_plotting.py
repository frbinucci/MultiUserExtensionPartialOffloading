import collections
import pickle
from decimal import Decimal

import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

def offloading_decisions_hist(dir,V_array,reg_start,device_selector):

    offloading = np.zeros(len(device_selector))
    not_offloading = np.zeros(len(device_selector))
    for v in V_array:
        filename = format_e(Decimal(v))
        with open(dir + f"simulation_{filename}.dat", 'rb') as config_dictionary_file:
            vx = pickle.load(config_dictionary_file)
            for i in device_selector:
                offloading_decisions = getattr(vx, 'offloading_decision')

                offloading_decisions = offloading_decisions[device_selector[i],reg_start:]

                offloading[i] = (np.count_nonzero(offloading_decisions)/offloading_decisions.size)*100
                not_offloading[i] = ((offloading_decisions.size-np.count_nonzero(offloading_decisions))/offloading_decisions.size)*100

        plt.bar(np.array(device_selector) + 0.00,offloading,width=0.25,color='b')
        plt.bar(np.array(device_selector) + 0.25,not_offloading,width=0.25,color='r')

        plt.title("Offloading Decisions per device")
        plt.legend(labels=['Offloading', 'Local'])
        plt.xlabel('Device Index')
        plt.ylabel('%')
        plt.grid()
        plt.show()



def plot_functions(dir, V_array,reg_start,type,**kwargs):

    qtot_arr = np.zeros(len(V_array))
    energy_arr = np.zeros(len(V_array))
    accuracy_array = np.zeros(len(V_array))

    reg_end = kwargs.get('reg_end',np.ones(len(V_array)).astype(int)*-1)
    label = kwargs.get('label','No Label')
    linestyle = kwargs.get('linestyle','solid')
    i=0
    for v in V_array:
        filename = format_e(Decimal(v))
        with open(dir + f"simulation_{filename}.dat", 'rb') as config_dictionary_file:
            vx = pickle.load(config_dictionary_file)
            compression_energy_array = getattr(vx,'compression_energy_array')
            transmission_energy_array = getattr(vx,'transmission_energy_array')
            server_energy_array = getattr(vx,'server_energy_array')
            accuracy = getattr(vx,'accuracy_status')
            #tot_energy = getattr(vx,'energy_array')
            qtot_array=getattr(vx,'qtot')

            mean_arrivals = kwargs.get('mean_arrivals',1)
            time_slot = kwargs.get('time_slot',50e-3)
            if reg_end[i]==-1:
                reg_end[i] = 10000


            accuracy = accuracy[:,reg_start[i]:reg_end[i]]
            accuracy = accuracy[accuracy!=0]
            device_energy = transmission_energy_array[:,reg_start[i]:reg_end[i]] + compression_energy_array[:,reg_start[i]:reg_end[i]]
            device_energy = np.sum(device_energy,axis=0)
            device_energy = np.mean(device_energy)

            if len(server_energy_array.shape)==1:
                server_energy = server_energy_array[reg_start[i]:reg_end[i]]
            else:
                server_energy = server_energy_array[0,reg_start[i]:reg_end[i]]


            cumulative_energy = device_energy+server_energy
            cumulative_energy = np.mean(cumulative_energy)

            qtot_array = qtot_array[:,reg_start[i]:reg_end[i]]

            if type=='device':
                device_energy = transmission_energy_array[:, reg_start[i]:reg_end[i]:] + compression_energy_array[:,reg_start[i]:reg_end[i]:]
                device_energy = np.sum(device_energy, axis=0)
                device_energy = np.mean(device_energy)
                energy_arr[i] = device_energy
            elif type=='transmission':
                device_energy = transmission_energy_array[:, reg_start[i]:reg_end[i]:]
                device_energy = np.sum(device_energy, axis=0)
                device_energy = np.mean(device_energy)
                energy_arr[i] = device_energy
            elif type=='compression':
                device_energy = compression_energy_array[:, reg_start[i]:reg_end[i]:]
                device_energy = np.sum(device_energy, axis=0)
                device_energy = np.mean(device_energy)
                energy_arr[i] = device_energy
            elif type=='server':
                tot_energy = server_energy
                energy_arr[i] = tot_energy
            elif type=='cumulative':
                tot_energy = cumulative_energy
                energy_arr[i] = tot_energy
            elif type=='latency':
                qtot = qtot_array.mean()
                qtot /= (mean_arrivals * 1 / time_slot)
                qtot_arr[i] = qtot
            elif type=='accuracy':
                accuracy = accuracy.mean()
                accuracy_array[i] = accuracy
        i+=1

    if type=='latency':
        plt.plot(V_array,qtot_arr,marker='v',label=label,linestyle=linestyle)
    elif type=='accuracy':
        plt.plot(V_array,accuracy_array,marker='v',label=label,linestyle=linestyle)
    else:
        plt.plot(V_array,energy_arr,marker='v',label=label,linestyle=linestyle)



def plot_energy_latency_trade_off(dir,V_array,reg_start,type,**kwargs):
    qtot_arr = np.zeros(len(V_array))
    energy_arr = np.zeros(len(V_array))
    linestyle = kwargs.get('linestyle',"solid")
    reg_end = kwargs.get('reg_end',np.ones(len(V_array)).astype(int)*-1)
    label = kwargs.get('label',None)

    i=0
    for v in V_array:
        filename = format_e(Decimal(v))
        with open(dir + f"simulation_{filename}.dat", 'rb') as config_dictionary_file:
            vx = pickle.load(config_dictionary_file)
            compression_energy_array = getattr(vx,'compression_energy_array')
            transmission_energy_array = getattr(vx,'transmission_energy_array')
            server_energy_array = getattr(vx,'server_energy_array')
            device_freq_array = getattr(vx,'server_frequencies_array')


            #tot_energy = getattr(vx,'energy_array')
            qtot_array=getattr(vx,'qtot')

            mean_arrivals = kwargs.get('mean_arrivals',1)
            time_slot = kwargs.get('time_slot',50e-3)

            if reg_end[i]==-1:
                reg_end[i] = compression_energy_array.size-1

            if type=='device' or type=='cumulative':
                device_energy = transmission_energy_array[:,reg_start[i]:reg_end[i]:] + compression_energy_array[:,reg_start[i]:reg_end[i]:]
                print(np.count_nonzero(device_energy))
            elif type=='transmission':
                device_energy = transmission_energy_array[:,reg_start[i]:reg_end[i]:]
            elif type == 'compression':
                device_energy = compression_energy_array[:, reg_start[i]:reg_end[i]:]

            if type!='server':
                device_energy = np.sum(device_energy, axis=0)
                device_energy = np.mean(device_energy)

                tot_energy = device_energy

            if len(server_energy_array.shape)==1:
                server_energy = server_energy_array[reg_start[i]:reg_end[i]]
            else:
                server_energy = server_energy_array[0,reg_start[i]:reg_end[i]]


            qtot_array = qtot_array[:,reg_start[i]:reg_end[i]]
            if type=='server':
                tot_energy =  server_energy.mean()
            elif type=='cumulative':
                tot_energy = (server_energy+device_energy).mean()

            qtot = qtot_array.mean()

            qtot/=(mean_arrivals*1/time_slot)

            energy_arr[i] = tot_energy
            qtot_arr[i] = qtot
        i+=1

    plt.plot(energy_arr,qtot_arr,marker='v',label=label,linestyle=linestyle)


def plot_virtual_queues(dir,V,queue):
    filename = format_e(Decimal(V))

    with open(dir + f"simulation_{filename}.dat", 'rb') as config_dictionary_file:
            tile=None
            vx = pickle.load(config_dictionary_file)
            if queue=='latency_virtual':
                title = 'Latency virtual queue'
                queue_to_plot = getattr(vx, 'latency_virtual_queue')
            elif queue=='accuracy_virtual':
                title = 'Accuracy virtual queue'
                queue_to_plot = getattr(vx, 'accuracy_virtual_queue')
                accuracy = getattr(vx,'accuracy_status')
                #accuracy=accuracy[accuracy!=0]
                accuracy = accuracy.flatten()
                print(np.mean(accuracy))
                print(collections.Counter(accuracy))
            elif queue=='qtot':
                title ='Total delay'
                queue_to_plot = getattr(vx, 'qtot')


            plt.title(f'V={V} {title}')
            n_plots = queue_to_plot.shape[0]
            for n in range(n_plots):
                plt.plot(queue_to_plot[n])

            plt.grid()
            plt.xlabel('Iteration index')
            plt.ylabel('Virtual queue')
            plt.show()

def plot_tradeoff_driver(base_dir,v_array,transient_matrix,accuracy_array,type,**kwargs):
    plt.gca().set_prop_cycle(None)
    xmin=kwargs.get('xmin',1e-3)
    xmax=kwargs.get('xmax',1)
    latency_constraint = kwargs.get('latency_constraint',150e-3)
    plot_constraint = kwargs.get('plot_constraint',False)
    linestyle = kwargs.get('linestyle',"solid")
    plot = kwargs.get('plot',True)
    reg_end_matrix = kwargs.get('reg_end_matrix',1e4*np.ones((len(accuracy_array),len(v_array))))
    custom_text = kwargs.get('custom_text','')
    plot_label=kwargs.get('plot_label',True)
    i=0
    print(reg_end_matrix)
    for acc in accuracy_array:
        label = None
        if plot_label==True:
            label = fr'$Accuracy\geq{acc}$% {custom_text}'
        plot_energy_latency_trade_off(f'{base_dir}/ACCURACY_{acc}/',v_array,transient_matrix[i,:],type,label=label,linestyle=linestyle,reg_end=reg_end_matrix[i:,][0])
        i+=1

    if type=='compression':
        plt.xlabel('Average Compression Energy per Time Slot (J)')
    elif type=='transmission':
        plt.xlabel('Average Transmission Energy per Time Slot (J)')
    elif type=='device':
        plt.xlabel('Average Device Energy consumption per Time Slot (J)')
    elif type=='server':
        plt.xlabel('Average Server Energy consumption per Time Slot (J)')
    elif type=='cumulative':
        plt.xlabel('Average Comulative Energy consumption per Time Slot (J)')
    else:
        raise Exception("Unknown tradeoff")
    if plot_constraint==True:
        plt.hlines(xmin=xmin, xmax=xmax, color='black', linestyles='dashed', label='Latency constraint', y=latency_constraint)
    plt.grid()
    plt.title('MU-mERAS Energy/Latency trade off')
    plt.ylabel('Latency (s)')
    #plt.xscale('log')
    plt.legend()
    if plot==True:
        plt.show()

def plot_functions_driver(base_dir,v_array,transient_matrix,accuracy_array,type,**kwargs):
    plt.gca().set_prop_cycle(None)
    xmin=kwargs.get('xmin',1e3)
    xmax=kwargs.get('xmax',1e8)
    constraint = kwargs.get('constraint',150e-3)
    plot_constraint = kwargs.get('plot_constraint',False)
    plot = kwargs.get('plot',True)
    custom_text = kwargs.get('custom_text','')
    linestyle = kwargs.get('linestyle','solid')
    plot_label=kwargs.get('plot_label',True)
    i=0
    for acc in accuracy_array:
        label=None
        if plot_label==True:
            label = fr'$Accuracy\geq{acc} {custom_text}$'
        plot_functions(f'{base_dir}/ACCURACY_{acc}/',v_array,transient_matrix[i,:],type,label=label,linestyle=linestyle)
        if type=='accuracy':
            if plot_constraint == True:
                plt.hlines(xmin=xmin, xmax=xmax, color='black', linestyles='dashed',
                           y=float(acc / 100))
                #print(f'Accuracy: {acc / 100}')
        i+=1

    if type=='compression':
        plt.ylabel('Average Compression Energy per Time Slot (J)')
    elif type=='transmission':
        plt.ylabel('Average Transmission Energy per Time Slot (J)')
    elif type=='device':
        plt.ylabel('Average Device Energy consumption per Time Slot (J)')
    elif type=='server':
        plt.ylabel('Average Server Energy consumption per Time Slot (J)')
    elif type=='cumulative':
        plt.ylabel('Average Comulative Energy consumption per Time Slot (J)')
    elif type=='accuracy':
        plt.ylabel('Average Accuracy per Time Slot (J)')
    elif type=='latency':
        plt.ylabel('Average Latency per Time Slot (J)')
    else:
        raise Exception("Unknown tradeoff")



    if type=='latency' and plot_constraint==True:
        plt.hlines(xmin=xmin, xmax=xmax, color='black', linestyles='dashed', label=f'{type} constraint', y=constraint)
    plt.grid()
    plt.title('MU-mERAS Energy/Latency trade off')
    plt.xlabel('Parameter V')
    plt.xscale('log')
    plt.legend()
    if plot == True:
        plt.show()

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def main():

    v_array = [1e8]
    for v in v_array:
        plot_virtual_queues('../GOOD_CHANNEL_OFFLOADING/ACCURACY_95/',v,'latency_virtual')
        plot_virtual_queues('../GOOD_CHANNEL_OFFLOADING/ACCURACY_95/',v,'accuracy_virtual')
    transient_matrix = np.zeros((3,7)).astype(int)
    regend_matrix = np.zeros((3,7)).astype(int)

    transient_matrix[0, :] = np.array([800, 800, 800, 1900, 1962, 8000, 8000])
    transient_matrix[1, :] = np.array([8000, 8000, 9500, 9200, 9000, 8000, 8000])
    transient_matrix[2, :] = np.array([8000, 8000, 8000, 6000, 8000, 8000, 8000])


    regend_matrix[0, :] = np.array([1000, 1000, 1000, 2000, 2000, 10000, 10000])
    regend_matrix[1, :] = np.array([10000, 10000, 10000, 10000,  10000, 10000, 10000])
    regend_matrix[2, :] = np.array([10000, 10000, 10000, 10000, 10000, 10000, 10000])

    plot_tradeoff_driver('../GOOD_CHANNEL_OFFLOADING', [1e3,1e4,1e5,1e6,1e7], transient_matrix, [95], 'cumulative',
                         xmin=0, xmax=0.05, plot_constraint=True,reg_end_matrix=regend_matrix,plot=False)

    '''plot_tradeoff_driver('../GOOD_CHANNEL', [1e3,1e4,1e5,1e6,1e7,1e8], transient_matrix, [70,90,95], 'device',
                         xmin=0, xmax=0.05, plot_constraint=True,reg_end_matrix=regend_matrix,plot=False,linestyle='solid')'''

    '''plot_functions_driver('../GOOD_CHANNEL', [1e3, 1e4, 1e5, 1e6, 1e7, 1e8], transient_matrix, [70, 90, 95], 'latency',
                          plot_constraint=True, linestyle='solid', plot_label=True)'''

    #plt.grid()
    transient_matrix = np.zeros((3,7)).astype(int)
    regend_matrix = np.zeros((3,7)).astype(int)

    transient_matrix[0, :] = np.array([8000, 8000, 8000, 9400, 8000, 9150, 9000])
    #transient_matrix[1, :] = np.array([9000, 9000, 9000, 9000, 8000, 9900, 9000])
    transient_matrix[1, :] = np.array([9000, 9000, 9000, 8500, 800, 9800, 9000])
    transient_matrix[2, :] = np.array([9000, 9000, 9000, 9000, 9500, 8100, 9000])

    regend_matrix[0, :] = np.array([10000, 10000, 10000, 10000, 10000, 10000, 10000])
    #regend_matrix[1, :] = np.array([10000, 10000, 10000, 10000, 10000, 10000, 10000])
    regend_matrix[1, :] = np.array([10000, 10000, 10000, 10000, 10000, 10000, 10000])
    regend_matrix[2, :] = np.array([10000, 10000, 10000, 10000, 10000, 10000, 10000])

    '''plot_tradeoff_driver('../BAD_CHANNEL/', [1e3,1e4,1e5,1e6,1e7,1e8], transient_matrix, [70,90,95], 'device',
                         xmin=0, xmax=0.4, plot_constraint=False,reg_end_matrix=regend_matrix,plot=False,linestyle='dotted',plot_label=False)'''




    '''plot_functions_driver('../BAD_CHANNEL/', [1e4,1e5,1e6,1e7,1e8],transient_matrix,[70,90,95],'transmission',plot_constraint=True,plot=False,linestyle='dotted',plot_label=False)

    plot_functions_driver('../BAD_OFFLOADING/', [1e4, 1e5, 1e6, 1e7, 1e8], transient_matrix, [70, 90, 95],
                          'transmission', plot_constraint=True,plot=False)'''
    plt.grid()
    plt.show()


    offloading_decisions_hist('../GOOD_CHANNEL_OFFLOADING/ACCURACY_95/',[1e3],1960,[0,1,2,3,4])

if __name__=='__main__':
    main()