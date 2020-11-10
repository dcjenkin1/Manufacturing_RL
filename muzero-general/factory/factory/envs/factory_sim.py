
import simpy
from collections import namedtuple, Counter
from itertools import count, filterfalse
import random
import math
import json

####################################################
########## CREATING THE WAFER CLASS  ###############
####################################################
class wafer_box(object):
    def __init__(self, sim_inst, HT, wafer_index, lead_dict, seq=0):
        self.env = sim_inst.env
        self.name = f"w{wafer_index}"
        self.start_time = sim_inst.env.now
        # self.number_wafers = number_wafers
        self.HT = HT
        self.seq = seq
        self.due_time = self.start_time + lead_dict[self.HT]*0.1
        self.queue_start_time = sim_inst.env.now


####################################################
########## CREATING THE MACHINE CLASS ##############
####################################################
class Machine(object):
    def __init__(self, sim_inst, name, station, break_mean=None, repair_mean=None, seed=None):
        self.env = sim_inst.env
        self.name = name
        self.station = station
        self.available = True
        self.broken = False
        self.wafer_being_proc = None
        self.parts_made = 0
        self.break_mean = break_mean
        self.random_generator = random.Random()
        if seed is not None:
            self.random_generator.seed(seed)

        if break_mean is not None:
            self.time_to_fail = self.time_to_failure()

        self.process = None
        self.repair_mean = repair_mean
        self.total_operational_time = 0
        self.takt_times = []
            

    def time_to_failure(self):
        """Return time until next failure for a machine."""
        return self.random_generator.expovariate(1/self.break_mean)

    def time_to_repair(self):
        """Return time until repair for a machine."""
        return self.random_generator.expovariate(1/self.repair_mean)

    def break_machine(self):
        """Break the machine after break_time"""
        assert not self.broken
        start = self.env.now
        try:
            yield self.env.timeout(self.time_to_fail)
            self.process.interrupt()
            self.time_to_fail = self.time_to_failure()
        except:
            self.time_to_fail -= self.env.now-start


    def part_process(self, wafer, sim_inst):
        # This function defines a process where a part of head type HT and sequence step seq is processed on the machine

        # get the amount of time it takes for the operation to run
        proc_t = sim_inst.get_proc_time(wafer.HT, wafer.seq)

        done_in = proc_t

        start_time = sim_inst.env.now

        sim_inst.ht_seq_wait[(wafer.HT, wafer.seq)].append(sim_inst.env.now-wafer.queue_start_time)
        while done_in:
            try:
                if self.break_mean is not None:
                    break_process = self.env.process(self.break_machine())
                start = self.env.now
                # print("started processing wafer %s on machine %s at %s"%(wafer.name, self.name, start))
                # wait until the process is done
                yield sim_inst.env.timeout(done_in)
                # set the wafer being processed to None
                self.wafer_being_proc = None
                # set machine to be available to process part
                self.available = True
                # print("Completed the process step of wafer %s on machine %s at %s and sent to "
                #       "next machine."%(wafer.name, self.name, self.env.now))
                # set the wafer to be at the next step in the sequence
                wafer.seq += 1
                # if seq is not the last sequence step then find the next station and choose actions for each of the
                # available machines in that station
                if wafer.seq < (len(sim_inst.recipes[wafer.HT])):
                    # add the part to the corresponding queue for the next operation in the sequence
                    sim_inst.queue_lists[sim_inst.recipes[wafer.HT][wafer.seq][0]].append(wafer)
                    # ts = sim_inst.recipes[wafer.HT][wafer.seq][0]
                    wafer.queue_start_time = sim_inst.env.now
                    sim_inst.arrival_times[sim_inst.recipes[wafer.HT][wafer.seq][0]].append(sim_inst.env.now)
                    sim_inst.n_HT_seq[wafer.HT][wafer.seq] += 1
                else:
                    # # add the part to the list of completed parts
                    # sim_inst.queue_lists['complete'].append(wafer)
                    sim_inst.cycle_time.append(self.env.now - wafer.start_time)
                    # print("Finished processing wafer %s at %s"%(wafer.name, self.env.now))
                    sim_inst.complete_wafer_dict[wafer.HT] += 1

                    sim_inst.lateness.append(max([0, (sim_inst.env.now-wafer.due_time)]))
                    # Update the due_wafers dictionary to indicate that wafers of this head type were completed

                    # Find the index of the earliest week for which there are one or more wafers of the given head type
                    # due.
                    week_index = next((i for i, x in enumerate(sim_inst.due_wafers[wafer.HT]) if x), None)

                    # Subtract wafer,number_wafers wafers from the corresponding list element
                    sim_inst.due_wafers[wafer.HT][week_index] -= 1

                    if all((sum(sim_inst.due_wafers[ht]) <= (sim_inst.n_part_mix-1)*sim_inst.part_mix[ht])
                           for ht in sim_inst.recipes.keys()):
                        sim_inst.order_completed = True
                        sim_inst.t_between_completions.append(sim_inst.env.now-sim_inst.order_complete_time)
                        sim_inst.order_complete_time = sim_inst.env.now
                        for ht in sim_inst.recipes.keys():
                            for i in range(sim_inst.part_mix[ht]):
                                new_wafer = wafer_box(sim_inst, ht, sim_inst.wafer_index,
                                                      sim_inst.lead_dict)
                                sim_inst.queue_lists[sim_inst.recipes[ht][0][0]].append(new_wafer)
                                sim_inst.n_HT_seq[ht][0] += 1
                                lead_time = sim_inst.lead_dict[ht]
                                total_processing_time = new_wafer.start_time + lead_time
                                week_number = int(total_processing_time / (7 * 24 * 60))
                                sim_inst.due_wafers[ht][week_number] += 1
                                sim_inst.wafer_index += 1

                if self.break_mean is not None:
                    break_process.interrupt()
                done_in = 0

            except simpy.Interrupt:
                self.broken = True
                done_in -= self.env.now - start
                yield self.env.timeout(self.time_to_repair())
                self.broken = False

        # Parts completed by this machine
        self.parts_made += 1
        self.total_operational_time += proc_t
        self.takt_times.append(sim_inst.env.now - start_time)

    def get_allowed_actions(self, sim_inst):
        #find all (HT, seq) tuples with non zero queues at the station of this machine
        # newlist = sorted([(ht,seq) for (ht, seq) in sim_inst.station_HT_seq[self.station] if sim_inst.n_HT_seq[ht][seq] > 0])
        # assert newlist == sorted(list(set((wafer.HT, wafer.seq) for wafer in sim_inst.queue_lists[self.station])))
        return sorted([(ht,seq) for (ht, seq) in sim_inst.station_HT_seq[self.station] if sim_inst.n_HT_seq[ht][seq] > 0])

    def get_allowed_ht(self, sim_inst):
        #find all HT with non zero queues at the station of this machine
        # newlist = sorted([(ht,seq) for (ht, seq) in sim_inst.station_HT_seq[self.station] if sim_inst.n_HT_seq[ht][seq] > 0])
        # assert newlist == sorted(list(set((wafer.HT, wafer.seq) for wafer in sim_inst.queue_lists[self.station])))
        return sorted([ht for ht in sim_inst.recipes.HT if sim_inst.n_HT_seq[ht][seq] > 0])

####################################################
########## CREATING THE FACTORY CLASS ##############
####################################################
class FactorySim(object):
    #Initialize simpy environment and set the amount of time the simulation will run for
    def __init__(self, sim_time, m_dict, recipes, lead_dict, part_mix, n_part_mix, path_to_wait_times=None, break_mean=None,
                 repair_mean=None, seed=None):
        self.break_mean = break_mean
        self.repair_mean = repair_mean
        self.order_completed = False
        self.allowed_actions = None
        self.env = simpy.Environment()
        self.Sim_time = sim_time
        self.next_machine = None
        # self.dgr = dgr_dict
        self.lead_dict = lead_dict
        # self.num_wafers = wafers_per_box
        self.part_mix = part_mix
        self.n_part_mix = n_part_mix
        # self.machine_failure = False
        self.lateness = []
        self.t_between_completions = []
        self.cumulative_reward = 0.
        self.cumulative_reward_list = []
        self.random_generator = random.Random()
        if seed is not None:
            self.random_generator.seed(seed)

        if path_to_wait_times is not None:
            with open(path_to_wait_times, 'r') as fp:
                self.wait_times = json.load(fp)
        else:
            self.wait_times = None

        # print(self.wait_times)

        # Number of future weeks we want to look into for calculating due dates
        self.FUTURE_WEEKS = math.ceil((max(lead_dict.values()) + sim_time) / (7*24*60) + 1)

        # Initialize an index that will be used to name each wafer box
        self.wafer_index = 0

        # Dictionary where the key is the name of the machine and the value is the station
        self.machine_dict = m_dict

        self.machines_list = [Machine(self, mach[0], mach[1], self.break_mean, self.repair_mean, seed=self.random_generator.randint(0,int(1e9))) for mach in self.machine_dict.items()]

        # create a list of all the station names
        self.stations = sorted(list(set(list(self.machine_dict.values()))))

        self.arrival_times = {station: [] for station in self.stations}

        self.station_machines = {station: [machine for machine in self.machines_list if machine.station == station] for station in self.stations}
        # ts = self.station_machines
        # sim_inst.recipes give the sequence of stations that must be processed at for the wafer of that head type to be completed
        self.recipes = recipes

        # create a list to store the number of complete wafers for each head type
        self.complete_wafer_dict = {}
        for ht in self.recipes.keys():
            d = {ht:0}
            self.complete_wafer_dict.update(d)

        self.number_of_machines = len(self.machine_dict)

        # Create a dictionary which holds lists that will contain 
        # the queues of wafer_box objects at each station and that have been completed
        self.queue_lists = {station: [] for station in self.stations}
        # self.queue_lists['complete'] = []

        self.order_complete_time = 0
        self.cycle_time = []
        self.step_reward = 0.

        # Create a dictionary which holds the number of wafers due in a given week of each head type
        self.due_wafers = {}
        for ht in self.recipes.keys():
            list_of_wafers_due_each_week = [0]*self.FUTURE_WEEKS
            d = {ht:list_of_wafers_due_each_week}
            self.due_wafers.update(d)

        # Creates a dictionary where the key is the toolset name and the value is a list of tuples of all head type and
        # sequence step combinations which may be processed at that station
        self.station_HT_seq = {station: [] for station in self.stations}

        for HT in self.recipes.keys():
            for seq, step in enumerate(self.recipes[HT]):
                self.station_HT_seq[step[0]].append((HT, seq))

        self.n_HT_seq = {ht: [0]*len(self.recipes[ht]) for ht in list(self.recipes.keys())}

        self.ht_seq_wait = dict()
        for HT in self.recipes.keys():
            for seq, step in enumerate(self.recipes[HT]):
                self.ht_seq_wait[(HT, seq)] = []

    def get_proc_time(self, ht, seq):
        proc_step = self.recipes[ht][seq]
        return proc_step[1]

    def get_rem_shop_time(self, ht, seq):
        steps = self.recipes[ht]
        n_steps = len(steps)

        rem_shop_t = 0
        if self.wait_times is not None:
            for i in range(seq, n_steps):
                rem_shop_t = rem_shop_t + self.get_proc_time(ht, i) + self.wait_times[str((ht, seq))]
        else:
            for i in range(seq, n_steps):
                rem_shop_t = rem_shop_t + self.get_proc_time(ht, i)

        # assert(rem_shop_t>0)
        return rem_shop_t



    def start(self):
        for ht in self.part_mix.keys():
            for i in range(self.n_part_mix*self.part_mix[ht]):
                new_wafer = wafer_box(self, ht, self.wafer_index, self.lead_dict)
                self.queue_lists[self.recipes[ht][0][0]].append(new_wafer)
                self.n_HT_seq[ht][0] += 1
                lead_time = self.lead_dict[ht]
                total_processing_time = new_wafer.start_time + lead_time
                week_number = int(total_processing_time / (7*24*60))
                self.due_wafers[ht][week_number] += 1
                self.wafer_index += 1

        for station in self.stations:
            if len(self.queue_lists[station])>0:
                for machine in self.station_machines[station]:
                    if machine.available:
                        allowed_actions = machine.get_allowed_actions(self)
                        if len(allowed_actions) > 0:
                            self.next_machine = machine
                            self.allowed_actions = allowed_actions
                            return
        while True:
            self.env.step()
            for station in self.stations:
                if len(self.queue_lists[station]) > 0:
                    for machine in self.station_machines[station]:
                        if machine.available:
                            allowed_actions = machine.get_allowed_actions(self)
                            if len(allowed_actions) > 0:
                                self.next_machine = machine
                                self.allowed_actions = allowed_actions
                                return


    def run_action(self, machine, wafer_choice):
        self.order_completed = False
        self.step_reward = 0.
        # Set the machine to be unavailable to process parts because it is now busy
        assert machine.available
        machine.available = False
        # Find the wafer that has that HT and seq
        # wafer_choice = next(wafer for wafer in self.queue_lists[machine.station] if wafer.HT == ht and wafer.seq == seq)
        # set the wafer being processed on this machine to wafer_choice
        machine.wafer_being_proc = wafer_choice
        # Remove the part from it's queue
        self.queue_lists[machine.station].remove(wafer_choice)
        self.n_HT_seq[wafer_choice.HT][wafer_choice.seq] -= 1
        # Begin processing the part on the machine
        machine.process = self.env.process(machine.part_process(wafer_choice, self))

        for station in self.stations:
            if len(self.queue_lists[station]) > 0:
                for machine in self.station_machines[station]:
                    if machine.available:
                        allowed_actions = machine.get_allowed_actions(self)
                        if len(allowed_actions) > 0:
                            self.next_machine = machine
                            self.allowed_actions = allowed_actions
                            self.cumulative_reward += self.step_reward
                            self.cumulative_reward_list.append(self.cumulative_reward)
                            return
        while True:
            before_time = self.env.now
            self.env.step()
            time_change = self.env.now-before_time
            current_week = math.ceil(self.env.now / (7 * 24 * 60))  # Calculating the current week
            for key, value in self.due_wafers.items():
                buffer_list = []  # This list stores value of previous unfinished wafers count
                buffer_list.append(sum(value[:current_week]))
                self.step_reward -= time_change*sum(buffer_list)

            for station in self.stations:
                if len(self.queue_lists[station]) > 0:
                    for machine in self.station_machines[station]:
                        if machine.available:
                            allowed_actions = machine.get_allowed_actions(self)
                            if len(allowed_actions) > 0:
                                self.next_machine = machine
                                self.allowed_actions = allowed_actions
                                self.cumulative_reward += self.step_reward
                                self.cumulative_reward_list.append(self.cumulative_reward)
                                return





