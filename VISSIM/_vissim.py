import numpy as np
import pandas as pd
import copy
import win32com.client as com


class vissim():
    def __init__(self,Filename):
        self.Vissim = com.Dispatch("Vissim.Vissim")
        self.filename = Filename

        self.warmup_time = 1800 # seconds
        self.time = 0
        self.simulation_time = 12600 # seconds
        self.control_cycle = 30 # make new action every 30s
        self.reward_interval = 1 # collect reward every 1mins

        self.state_collector = {'downstream':(10,21,3),'upstream':(10,10,3),'buslane':(10,10,3),'metering':(1,2,1)} # (minutes,vd count,feature) saves nn input shape
        self.vd_number = {'downstream':1,'upstream':16,'buslane':34} # collected vd startup number
        self.state = None
        self.previous_state = None

        # define vissim object number
        self.signal_controller = {'mainline':1,'ramp':2} # for signal action caontrol
        #self.mainline_q = {'30.1':23,'30.551':24,'30.97':25,'31.54':26,'32.12':27,'32.743':28,'33.65':29,'34.898':30,'36.073':31,'37.225':32,'42.359':33} # referrence get_mainline_queue
        self.mainline_q = {'0.5':23,'0.451':24,'0.419':25,'0.57':26,'0.58':27,'0.623':28,'0.907':29,'1.248':30,'1.175':31,'1.152':32,'5.134':33}
        self.bottleneck_mileage_p = {'16.196':2,'21.055':9,'26.705':17,'28.420':20} # referrence get_bottleneck_MPrate()
        self.tunnel_mileage_p = {'15.488':1}
        self.merging_mileage_p = {'29.600':44} # referrence get_merging_MPrate()
        self.ramp_q = [1,2] # referrence get_ramp_queue()
        self.tt_number = {'Yilan-Toucheng (Bus)':5,'Yilan-Toucheng (Car)':4,'Toucheng-Tunnel Entry (Bus)':3,'Toucheng-Tunnel Entry (Car)':2,'Tunnel Entry-Exit':1}
        self.metering_rate_collector = {'mainline':22,'ramp':45}
    
    # vissim base function
    def open(self): # open vissim interface
        self.Vissim.LoadNet(self.filename)
        # set simulation to quick mode
        self.Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode",1)

    def reset(self,seed): # reset environment
        self.time = 0
        self.state = None
        self.previous_state = None
        self.Vissim.Simulation.Stop() # stop simulation
        #set vissim random seed
        self.set_randseed(seed)
        # clear previous simulation results
        for simRun in self.Vissim.Net.SimulationRuns:
            self.Vissim.Net.SimulationRuns.RemoveSimulationRun(simRun)

    def warm_up(self): # env warm up
        self.time = self.warmup_time
        self.Vissim.Simulation.SetAttValue('SimBreakAt', 1)
        self.Vissim.Simulation.RunContinuous()
        for k,v in self.signal_controller.items():
            sc = self.Vissim.Net.SignalControllers.ItemByKey(v)
            sc.SGs.ItemByKey(1).SetAttValue("SigState",'OFF')
        self.Vissim.Simulation.SetAttValue('SimBreakAt', self.warmup_time)
        self.Vissim.Simulation.RunContinuous()

    def simulation(self,seed):
        self.set_randseed(seed)
        self.Vissim.Simulation.RunContinuous()

    def set_randseed(self, seed):# set random seed in vissim
        self.Vissim.Simulation.SetAttValue('RandSeed', seed)

    def create_vehcom(self,i,ref1,ref2,ref3):
        cartype = {0:self.Vissim.Net.VehicleTypes.ItemByKey(100),\
                1:self.Vissim.Net.VehicleTypes.ItemByKey(300),\
                2:self.Vissim.Net.VehicleTypes.ItemByKey(400)}
        reflow = [ref1,ref2,ref3]
        temp = [reflow.index(ref) for ref in reflow if ref != 0]
        if len(set(temp)) != len(temp):
            temp[2] = 2
        car = [cartype[ref] for ref in temp]
        for j in range(len(temp)):
            car.insert(j*2+1,self.Vissim.Net.DesSpeedDistributions.ItemByKey(40))

        self.Vissim.Net.VehicleCompositions.AddVehicleComposition(i,car)
        Rel_Flows = self.Vissim.Net.VehicleCompositions.ItemByKey(i).VehCompRelFlows.GetAll()
        for j in range(len(temp)):
            Rel_Flows[j].SetAttValue('RelFlow',reflow[temp[j]])

    '''old version of vehinput and vehroute setup
        def set_vehinput(self, vehinp):
            # input rdic must be a pd.DataFrame, which contain data as follow
            # ['起點車輛數','起點小客車比例','頭城入口車輛數','頭城入口小客車比例','頭城入口車輛數','頭城入口小客車比例','宜蘭入口車輛數','宜蘭入口小客車比例']
            # define vissim vehicle input number and name, name must match the input data column names
            vdic = {1:['起點車輛數','起點小客車比例'],2:['頭城入口車輛數','頭城入口小客車比例'],3:['頭城入口車輛數','頭城入口小客車比例'],4:['宜蘭入口車輛數','宜蘭入口小客車比例']}
            for k in vdic:
                i = vdic.get(k)
                for time in range(1,49):
                    self.Vissim.Net.VehicleInputs.ItemByKey(k).SetAttValue('Volume(%s)'%str(time),vehinp.loc[time-1,i[0]])
                    self.Vissim.Net.VehicleInputs.ItemByKey(k).SetAttValue('VehComp(%s)'%str(time),vehinp.loc[time-1,i[1]]*100)
        
        def set_vehroute(self, routeratio):
            # input rdic must be a pd.DataFrame, which contain data as follow 
            # ['頭城小客車未下匝道比例','頭城大客車未下匝道比例','宜蘭小客車未下匝道比例','宜蘭大客車未下匝道比例']
            # define vissim vehicle route number and name, name must match the input data column names
            rdic = {3:'頭城小客車未下匝道比例',4:'頭城大客車未下匝道比例',1:'宜蘭小客車未下匝道比例',2:'宜蘭大客車未下匝道比例'}
            for k in rdic:
                i = rdic.get(k)
                for time in range(1,49):
                    SVRD_number = k # SVRD = Static Vehicle Routing Decision
                    SVR_number = 1 # SVR = Static Vehicle Route (of a specific Static Vehicle Routing Decision)
                    new_relativ_flow = routeratio.loc[time-1,i]
                    # set keep drinving main line vehicle proportion
                    self.Vissim.Net.VehicleRoutingDecisionsStatic.ItemByKey(SVRD_number).VehRoutSta.ItemByKey(1)\
                        .SetAttValue('RelFlow(%s)'%str(time), new_relativ_flow)
                    # set leaving highway vehicle proportion
                    self.Vissim.Net.VehicleRoutingDecisionsStatic.ItemByKey(SVRD_number).VehRoutSta.ItemByKey(2)\
                        .SetAttValue('RelFlow(%s)'%str(time), 1 - new_relativ_flow)'''

    def set_vehinput(self,vehinp):
        # input rdic must be a pd.DataFrame, which contain data as follow
        # ['起點車輛數','起點小客車比例','頭城入口車輛數','頭城入口小客車比例','頭城入口車輛數','頭城入口小客車比例','宜蘭入口車輛數','宜蘭入口小客車比例']
        # define vissim vehicle input number and name, name must match the input data column names
        vdic = {1:['起點車輛數','起點小客車比例','起點大客車比例'],\
                2:['頭城入口車輛數','頭城入口小客車比例','頭城入口大客車比例'],\
                3:['頭城入口車輛數','頭城入口小客車比例','頭城入口大客車比例'],\
                4:['宜蘭入口車輛數','宜蘭入口小客車比例','宜蘭入口大客車比例']}
        idx = 101
        vip = self.Vissim.Net.VehicleInputs
        for k in vdic:
            i = vdic.get(k)
            for time in range(1,self.simulation_time//300):
                ref1 = vehinp.loc[time-1,i[1]]
                ref2 = vehinp.loc[time-1,i[2]]
                ref3 = 100-ref1-ref2
                if ref3 > 0:
                    print(ref1,ref2,ref3)
                    self.create_vehcom(idx,ref1,ref2,ref3)
                    vip.ItemByKey(k).SetAttValue('Volume(%s)'%str(time),vehinp.loc[time-1,i[0]])
                    vip.ItemByKey(k).SetAttValue('VehComp(%s)'%str(time),idx)
                    vip.ItemByKey(k).SetAttValue('VolType(%s)'%str(time),'Exact')
                    idx += 1
                elif ref2 == 0 and ref3 != 0:
                    print(ref1,ref2,ref3)
                    self.create_vehcom(idx,ref1,ref2,ref3)
                    vip.ItemByKey(k).SetAttValue('Volume(%s)'%str(time),vehinp.loc[time-1,i[0]])
                    vip.ItemByKey(k).SetAttValue('VehComp(%s)'%str(time),idx)
                    vip.ItemByKey(k).SetAttValue('VolType(%s)'%str(time),'Exact')
                    idx += 1
                else:
                    vip.ItemByKey(k).SetAttValue('Volume(%s)'%str(time),vehinp.loc[time-1,i[0]])
                    vip.ItemByKey(k).SetAttValue('VehComp(%s)'%str(time),ref1)
                    vip.ItemByKey(k).SetAttValue('VolType(%s)'%str(time),'Exact')

    def set_vehroute(self,routeratio):
        # input rdic must be a pd.DataFrame, which contain data as follow 
        # ['頭城小客車未下匝道比例','頭城大客車未下匝道比例','頭城聯結車未下匝道比例',\
        #'宜蘭小客車未下匝道比例','宜蘭大客車未下匝道比例','宜蘭聯結車未下匝道比例']
        # define vissim vehicle route number and name, name must match the input data column names
        rdic = {4:'頭城小客車未下匝道比例',5:'頭城大客車未下匝道比例',6:'頭城聯結車未下匝道比例',\
                1:'宜蘭小客車未下匝道比例',2:'宜蘭大客車未下匝道比例',3:'宜蘭聯結車未下匝道比例'}
        vr = self.Vissim.Net.VehicleRoutingDecisionsStatic
        for k in rdic:
            i = rdic.get(k)
            for time in range(1,self.simulation_time//300):
                SVRD_number = k # SVRD = Static Vehicle Routing Decision
                SVR_number = 1 # SVR = Static Vehicle Route (of a specific Static Vehicle Routing Decision)
                new_relativ_flow = routeratio.loc[time-1,i]
                # set keep drinving main line vehicle proportion
                vr.ItemByKey(SVRD_number).VehRoutSta.ItemByKey(1).SetAttValue('RelFlow(%s)'%str(time), new_relativ_flow)
                # set leaving highway vehicle proportion
                vr.ItemByKey(SVRD_number).VehRoutSta.ItemByKey(2).SetAttValue('RelFlow(%s)'%str(time), 1 - new_relativ_flow)

    def reset_driving_behavior(self,ind):
        section = {'front':[102,20002],'mid':[103,20003],'end':[104,20004]}
        pname = ["W99cc0","W99cc1Distr","W99cc2","W99cc7"]
        for pn in range(len(pname)):
            if pn == 0 :
                self.Vissim.Net.Links.ItemByKey(section['front'][0]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn])
                self.Vissim.Net.Links.ItemByKey(section['front'][1]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn])
                self.Vissim.Net.Links.ItemByKey(section['mid'][0]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn+4])
                self.Vissim.Net.Links.ItemByKey(section['mid'][1]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn+4])
                self.Vissim.Net.Links.ItemByKey(section['end'][0]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn+8])
                self.Vissim.Net.Links.ItemByKey(section['end'][1]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn+8])
            elif pn == 1:
                self.Vissim.Net.Links.ItemByKey(section['front'][0]).LinkBehavType.DrivBehavDef.W99cc1Distr.SetAttValue("Mean",ind[pn])
                self.Vissim.Net.Links.ItemByKey(section['front'][1]).LinkBehavType.DrivBehavDef.W99cc1Distr.SetAttValue("Mean",ind[pn])
                self.Vissim.Net.Links.ItemByKey(section['mid'][0]).LinkBehavType.DrivBehavDef.W99cc1Distr.SetAttValue("Mean",ind[pn+4])
                self.Vissim.Net.Links.ItemByKey(section['mid'][1]).LinkBehavType.DrivBehavDef.W99cc1Distr.SetAttValue("Mean",ind[pn+4])
                self.Vissim.Net.Links.ItemByKey(section['end'][0]).LinkBehavType.DrivBehavDef.W99cc1Distr.SetAttValue("Mean",ind[pn+8])
                self.Vissim.Net.Links.ItemByKey(section['end'][1]).LinkBehavType.DrivBehavDef.W99cc1Distr.SetAttValue("Mean",ind[pn+8])
            elif pn == 2 :
                self.Vissim.Net.Links.ItemByKey(section['front'][0]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn])
                self.Vissim.Net.Links.ItemByKey(section['front'][1]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn])
                self.Vissim.Net.Links.ItemByKey(section['mid'][0]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn+4])
                self.Vissim.Net.Links.ItemByKey(section['mid'][1]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn+4])
                self.Vissim.Net.Links.ItemByKey(section['end'][0]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn+8])
                self.Vissim.Net.Links.ItemByKey(section['end'][0]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn+8])
            elif pn == 3 :
                self.Vissim.Net.Links.ItemByKey(section['front'][0]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn])
                self.Vissim.Net.Links.ItemByKey(section['front'][1]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn])
                self.Vissim.Net.Links.ItemByKey(section['mid'][0]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn+4])
                self.Vissim.Net.Links.ItemByKey(section['mid'][1]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn+4])
                self.Vissim.Net.Links.ItemByKey(section['end'][0]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn+8])
                self.Vissim.Net.Links.ItemByKey(section['end'][0]).LinkBehavType.DrivBehavDef.SetAttValue(pname[pn],ind[pn+8])

    # state functions
    def update_state(self):
        self.previous_state = copy.deepcopy(self.state)
        self.state = self.get_state()
        self.state = self.state_normalization(self.state)

    def state_normalization(self,state):
        s = copy.deepcopy(state)
        for k in s:
            if k != 'metering':
                # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
                # https://aifreeblog.herokuapp.com/posts/54/data_science_203/
                s[k][:,:,0] *= 2/70  # flow range(0,350) since the general highway capacity is around 2100 veh/hr/lane (4200/60=70(veh/hr))
                s[k][:,:,0] += -1
                s[k][:,:,1] *= 2/120 # speed range(0,120)
                s[k][:,:,1] += -1
                s[k][:,:,2] *= 2/1 # occupancy range(0,100)
                s[k][:,:,2] += -1
            else:
                s[k][:,:,0] *= 2/70  # flow range(0,350) since the general highway capacity is around 2100 veh/hr/lane (4200/60=70(veh/hr))
                s[k][:,:,0] += -1
        return s

    def get_state(self):
        temp_state = {}
        
        # get vd count in self.state_collector for each section
        for k,v in self.state_collector.items():
            if k != 'metering':
                # 1800(s)//30(collect time interval) - (1800 - 600)//30 = 20
                current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - (1800 - v[0]*60 )//30
                past_interval = current_interval - 2*v[0] + 1 # 20(time interval) - 2(interval/min)*10(min) + 1 = 1 
                save = np.zeros((v[0], v[1], v[2]), dtype='float32')
                # run every vd
                idx = 0
                for j in range(self.vd_number[k], self.vd_number[k] + v[1]):
                    f,s,o,tidx = 0,0,0,1
                    dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(j)
                    # run every time interval
                    for i in range(past_interval, current_interval + 1):
                        veh_count = 0
                        # check every feature is None type or not, for flow and occ are set None to 0, and for speed is set to 90(km/hr)
                        temp = dc.AttValue("Vehs(Current,%i,All)"%i) or 0
                        veh_count += temp
                        f += temp
                        
                        temp = dc.AttValue("SpeedAvgHarm(Current,%i,All)"%i) or 90
                        s += temp*veh_count
                        
                        temp = dc.AttValue("OccupRate(Current,%i,All)"%i) or 0
                        o += temp
                        # save state every 1 min, since the time interval is set to 30s
                        if tidx%2 == 0:
                            save[tidx//2-1,idx,0] = f # *1.4
                            if f > 0:
                                save[tidx//2-1,idx,1] = s/f
                            save[tidx//2-1,idx,2] = o/2
                            f,s,o = 0,0,0
                        tidx += 1
                    idx += 1
                temp_state[k] = save
            else:
                # 1800(s)//30(collect time interval) - (1800 - 600)//30 = 20
                current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - (1800 - v[0]*60 )//30
                past_interval = current_interval - 2*v[0] + 1 # 20(time interval) - 2(interval/min)*10(min) + 1 = 1 
                save = np.zeros((v[0], v[1], v[2]), dtype='float32')
                # run every vd
                idx = 0
                for j in self.metering_rate_collector:
                    j = self.metering_rate_collector[j]
                    f,tidx = 0,1
                    dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(j)
                    # run every time interval
                    for i in range(past_interval, current_interval + 1):
                        # check every feature is None type or not, for flow and occ are set None to 0, and for speed is set to 90(km/hr)
                        temp = dc.AttValue("Vehs(Current,%i,All)"%i) or 0
                        f += temp
                        # save state every 1 min, since the time interval is set to 30s
                        if tidx%2 == 0:
                            save[tidx//2-1,idx,0] = f # *1.4
                            f = 0
                        tidx += 1
                    idx += 1
                temp_state[k] = save
        return temp_state

    def get_state_(self):
        temp_state = {}
        
        # get vd count in self.state_collector for each section
        for k,v in self.state_collector.items():
            if k != 'metering':
                # 1800(s)//30(collect time interval) - (1800 - 600)//30 = 20
                current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - (1800 - v[0]*60 )//30
                past_interval = current_interval - 2*v[0] + 1 # 20(time interval) - 2(interval/min)*10(min) + 1 = 1 
                save = np.zeros((v[0], v[1], v[2]), dtype='float32')
                # run every vd
                idx = 0
                for j in range(self.vd_number[k], self.vd_number[k] + v[1]):
                    f,s,o,tidx = 0,0,0,1
                    dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(j)
                    # run every time interval
                    for i in range(past_interval, current_interval + 1):
                        veh_count = 0
                        # check every feature is None type or not, for flow and occ are set None to 0, and for speed is set to 90(km/hr)
                        temp = dc.AttValue("Vehs(Current,%i,All)"%i) or 0
                        veh_count += temp
                        f += temp
                        
                        if dc.AttValue("OccupRate(Current,%i,All)"%i) >= 0.5 and dc.AttValue("SpeedAvgHarm(Current,%i,All)"%i) == None:
                            temp = 0
                            s += temp*veh_count
                        else:
                            temp = dc.AttValue("SpeedAvgHarm(Current,%i,All)"%i)
                            s += temp*veh_count
                        temp = dc.AttValue("OccupRate(Current,%i,All)"%i) or 0
                        o += temp
                        # save state every 1 min, since the time interval is set to 30s
                        if tidx%2 == 0:
                            save[tidx//2-1,idx,0] = f # *1.4
                            if f > 0:
                                save[tidx//2-1,idx,1] = s/f
                            save[tidx//2-1,idx,2] = o/2
                            f,s,o = 0,0,0
                        tidx += 1
                    idx += 1
                temp_state[k] = save
            else:
                # 1800(s)//30(collect time interval) - (1800 - 600)//30 = 20
                current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - (1800 - v[0]*60 )//30
                past_interval = current_interval - 2*v[0] + 1 # 20(time interval) - 2(interval/min)*10(min) + 1 = 1 
                save = np.zeros((v[0], v[1], v[2]), dtype='float32')
                # run every vd
                idx = 0
                for j in self.metering_rate_collector:
                    j = self.metering_rate_collector[j]
                    f,tidx = 0,1
                    dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(j)
                    # run every time interval
                    for i in range(past_interval, current_interval + 1):
                        # check every feature is None type or not, for flow and occ are set None to 0, and for speed is set to 90(km/hr)
                        temp = dc.AttValue("Vehs(Current,%i,All)"%i) or 0
                        f += temp
                        # save state every 1 min, since the time interval is set to 30s
                        if tidx%2 == 0:
                            save[tidx//2-1,idx,0] = f # *1.4
                            f = 0
                        tidx += 1
                    idx += 1
                temp_state[k] = save
        return temp_state

    # action functions
    def activate_signal(self,which='all'):# take signal control and set the initial phase as green
        if which == 'all':
            for k,v in self.signal_controller.items():
                sc = self.Vissim.Net.SignalControllers.ItemByKey(v)
                sc.SGs.ItemByKey(1).SetAttValue("SigState",'GREEN')
        elif which == 'mainline':
            sc = self.Vissim.Net.SignalControllers.ItemByKey(self.signal_controller['mainline'])
            sc.SGs.ItemByKey(1).SetAttValue("SigState",'GREEN')
            sc = self.Vissim.Net.SignalControllers.ItemByKey(self.signal_controller['ramp'])
            sc.SGs.ItemByKey(1).SetAttValue("ContrByCOM", False)

    def next_time_step(self, phase_time):# signal control by com
        if phase_time != 0:
            self.time += phase_time
            self.Vissim.Simulation.SetAttValue('SimBreakAt', self.time)
            self.Vissim.Simulation.RunContinuous()

    def metering1(self, actionM):
        M_closed = False
        amber = 3
        min_green = 4

        # action == 19 means all red in the next time step, 19+12>30
        if (actionM + 12 >= self.control_cycle):
            M_closed = True
        
        if M_closed:
            self.mainline_metering('R')
            self.next_time_step(self.control_cycle)
        else:
            green1 = actionM + min_green
            self.mainline_metering('G')
            self.next_time_step(green1)
            self.mainline_metering('Y')
            self.next_time_step(amber)
            self.mainline_metering('R')
            self.next_time_step(self.control_cycle - green1 - amber)

    def metering2(self,actionM,actionR):
        M_closed = False
        R_closed = False
        amber = 3
        min_green = 4

        # action == 19 means all red in the next time step, 19+12>30
        if (actionM + 12 >= self.control_cycle):
            M_closed = True
        if (actionR + 12 >= self.control_cycle):
            R_closed = True
        
        if M_closed and not(R_closed):
            green2 = actionR + min_green
            self.mainline_metering('R')
            self.ramp_metering('G')
            self.next_time_step(green2)
            self.ramp_metering('Y')
            self.next_time_step(amber)
            self.ramp_metering('R')
            self.next_time_step(self.control_cycle - green2 -amber)
        elif not(M_closed) and R_closed:
            green1 = actionM + min_green
            self.ramp_metering('R')
            self.mainline_metering('G')
            self.next_time_step(green1)
            self.mainline_metering('Y')
            self.next_time_step(amber)
            self.mainline_metering('R')
            self.next_time_step(self.control_cycle - green1 -amber)
        elif M_closed and R_closed:
            self.mainline_metering('R')
            self.ramp_metering('R')
            self.next_time_step(self.control_cycle)
        else:
            # minimum greeen time = 4s
            # minimum red time = 5s
            # action size = 19(0-18s)
            green1 = actionM + min_green
            green2 = actionR + min_green
            if green1 < green2: # mainline metering ends up first
                # mainline has 15s green time, ramp has 20s green time
                # the control flow will be
                #0->(G)(G)->run15s->(Y)(G)->run3s->(R)(G)
                #->stopat20->(R)(Y)->run3s->(R)(R)->stopat30
                #control_scheme = [green1, amber, green2-green1-amber, amber, self.control_cycle-green2-amber]
                
                self.mainline_metering('G')
                self.ramp_metering('G')
                self.next_time_step(green1)
                self.mainline_metering('Y')
                if green2 <= green1 + amber:
                    self.next_time_step(green2-green1)
                    self.ramp_metering('Y')
                    self.next_time_step(amber-(green2-green1))
                    self.mainline_metering('R')
                    self.next_time_step(amber-(amber-(green2-green1)))
                    self.ramp_metering('R')
                    self.next_time_step(self.control_cycle - green2 - amber)
                else:
                    self.next_time_step(amber)
                    self.mainline_metering('R')
                    self.next_time_step(green2-green1-amber)
                    self.ramp_metering('Y')
                    self.next_time_step(amber)
                    self.ramp_metering('R')
                    self.next_time_step(self.control_cycle-green2-amber)
            
            elif green2 < green1:
                self.mainline_metering('G')
                self.ramp_metering('G')
                self.next_time_step(green2)
                self.ramp_metering('Y')
                if green1 <= green2 + amber:
                    self.next_time_step(green1-green2)
                    self.mainline_metering('Y')
                    self.next_time_step(amber-(green1-green2))
                    self.ramp_metering('R')
                    self.next_time_step(amber-(amber-(green1-green2)))
                    self.mainline_metering('R')
                    self.next_time_step(self.control_cycle - green1 - amber)
                else:
                    self.next_time_step(amber)
                    self.ramp_metering('R')
                    self.next_time_step(green1-green2-amber)
                    self.mainline_metering('Y')
                    self.next_time_step(amber)
                    self.mainline_metering('R')
                    self.next_time_step(self.control_cycle-green1-amber)
            
            else:
                self.mainline_metering('G')
                self.ramp_metering('G')
                self.next_time_step(green1)
                self.mainline_metering('Y')
                self.ramp_metering('Y')
                self.next_time_step(amber)
                self.mainline_metering('R')
                self.ramp_metering('R')
                self.next_time_step(self.control_cycle-green1-amber)

    def mainline_metering(self,phase=None):
        sc = self.Vissim.Net.SignalControllers.ItemByKey(self.signal_controller['mainline'])
        if phase == 'G':
            sc.SGs.ItemByKey(1).SetAttValue("SigState", "GREEN")
            #print('mainline G',self.time)
        elif phase == 'Y':
            sc.SGs.ItemByKey(1).SetAttValue("SigState", "AMBER")
            #print('mainline Y',self.time)
        elif phase == 'R':
            sc.SGs.ItemByKey(1).SetAttValue("SigState", "RED")
            #print('mainline R',self.time)
        else:
            print('Wrong phase!')

    def ramp_metering(self,phase=None):
        sc = self.Vissim.Net.SignalControllers.ItemByKey(self.signal_controller['ramp'])
        if phase == 'G':
            sc.SGs.ItemByKey(1).SetAttValue("SigState", "GREEN")
            #print('ramp G',self.time)
        elif phase == 'Y':
            sc.SGs.ItemByKey(1).SetAttValue("SigState", "AMBER")
            #print('ramp Y',self.time)
        elif phase == 'R':
            sc.SGs.ItemByKey(1).SetAttValue("SigState", "RED")
            #print('ramp R',self.time)
        else:
            print('Wrong phase!')

    # reward functions
    def get_mainline_queue(self):
        #self.mainline_q = {'0.5':23,'0.451':24,'0.419':25,'0.57':26,'0.58':27,'0.623':28,'0.907':29,'1.248':30,'1.175':31,'1.152':32,'5.134':33}
        
        # 1800(s)//30(collect time interval) - (1800 - 600)//30 = 20
        current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - (1800 - 10*60 )//30 
        past_interval = int(current_interval - self.reward_interval*2 + 1) # 20(time interval) - 2(interval/min)*5(min) + 1 = 1 
        queue_count = np.zeros((current_interval-past_interval+1),dtype="float32")
        m=0
        for i in range(past_interval, current_interval + 1):
            speed = np.zeros((len(self.mainline_q)),dtype="float32")
            occ   = np.zeros((len(self.mainline_q)),dtype="float32")
            j = 0
            # run every vd
            for k,v in self.mainline_q.items():
                s,o = 0,0
                dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)

                # check every feature is None type or not, for flow and occ are set None to 0, and for speed is set to 90(km/hr)
                temp = dc.AttValue("SpeedAvgHarm(Current,%i,All)"%i) or 90
                s = temp

                temp = dc.AttValue("OccupRate(Current,%i,All)"%i) or 0
                o = temp

                speed[j] = s
                occ[j] = o
                j += 1
            #print(speed)
            #print(occ)

            for j,k,l in zip(speed,occ,self.mainline_q):       
                if j <= 40:
                    #print('flow %f speed %f occ %f'%(l,i,j))
                    queue_count[m]+=k*100*float(l)*2/0.965
                elif l == '1.248' and j <= 40:
                    queue_count[m] = 200*12.759
            m+=1
        if any(i >= 200*12.759 for i in queue_count):
            queue_count = 200*12.759
        else:
            queue_count = np.mean(queue_count)
        return queue_count
    
    def get_bottleneck_MPrate(self): # MP (mileage productioin)
        # 1800(s)//30(collect time interval) - (1800 - 600)//30 = 20
        current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - (1800 - 10*60 )//30
        past_interval = int(current_interval - self.reward_interval*2 + 1) # 20(time interval) - 2(interval/min)*5(min) + 1 = 1 
        mileage_production = np.zeros((len(self.bottleneck_mileage_p)),dtype="float32")
        j = 0
        # run every vd
        for k,v in self.bottleneck_mileage_p.items():
            s = 0
            dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range(past_interval, current_interval + 1):
                veh_count = 0
                # check every feature is None type or not, for flow and occ are set None to 0, and for speed is set to 90(km/hr)
                temp = dc.AttValue("Vehs(Current,%i,All)"%i) or 0
                veh_count = temp
                #print(temp)

                temp = dc.AttValue("SpeedAvgHarm(Current,%i,All)"%i) or 90
                #print(temp)
                s += temp*veh_count
            # save mileage production every 5 min
            mileage_production[j] = s
            j += 1
        return mileage_production

    def get_bottleneck_throughput(self):
        # 1800(s)//30(collect time interval) - (1800 - 600)//30 = 20
        current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - (1800 - 10*60)//30
        past_interval = int(current_interval - self.reward_interval*2 + 1) # 20(time interval) - 2(interval/min)*5(min) + 1 = 1 
        bottleneck_throughput = np.zeros((len(self.bottleneck_mileage_p)),dtype="float32")
        j = 0
        # run every vd
        for k,v in self.bottleneck_mileage_p.items():
            f = 0
            dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range(past_interval, current_interval + 1):
                # check every feature is None type or not, for flow and occ are set None to 0, and for speed is set to 90(km/hr)
                temp = dc.AttValue("Vehs(Current,%i,All)"%i) or 0
                f += temp

            bottleneck_throughput[j] = f
            j += 1
        return bottleneck_throughput

    def get_merging_MPrate(self): # merging MPrate is only collected for buses. MP (mileage productioin)
        # 1800(s)//30(collect time interval) - (1800 - 300)//30 = 10
        current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - int(1800 - 10*60 )//30
        past_interval = int(current_interval - self.reward_interval*2 + 1) # 10(time interval) - 2(interval/min)*5(min) + 1 = 1 
        mileage_production = np.zeros((len(self.merging_mileage_p)),dtype="float32")
        j = 0
        # run every vd
        for k,v in self.merging_mileage_p.items():
            s = 0
            dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range(past_interval, current_interval + 1):
                veh_count = 0
                # check every feature is None type or not, for flow and occ are set None to 0, and for speed is set to 90(km/hr)
                temp = dc.AttValue("Vehs(Current,%i,30)"%i) or 0
                veh_count = temp

                temp = dc.AttValue("SpeedAvgHarm(Current,%i,30)"%i) or 90
                s += temp*veh_count
            # save mileage production every 5 min
            mileage_production[j] = s
            j += 1
        return np.sum(mileage_production)

    def get_merging_throughput(self):
        # 1800(s)//30(collect time interval) - (1800 - 300)//30 = 10
        current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - int(1800 - 10*60 )//30
        past_interval = int(current_interval - self.reward_interval*2 + 1) # 10(time interval) - 2(interval/min)*5(min) + 1 = 1 
        merging_throughput = np.zeros((len(self.merging_mileage_p)),dtype="float32")
        j = 0
        # run every vd
        for k,v in self.merging_mileage_p.items():
            f = 0
            dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range(past_interval, current_interval + 1):
                # check every feature is None type or not, for flow and occ are set None to 0, and for speed is set to 90(km/hr)
                temp = dc.AttValue("Vehs(Current,%i,30)"%i) or 0
                f += temp

            # save mileage production every 5 min
            merging_throughput[j] = f
            j += 1 
        return np.sum(merging_throughput)

    def get_ramp_queue(self):
        # 1800(s)//60(collect time interval) - (1800 - 5*60)//60 = 5
        current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle*0.5)) - 1500//60
        past_interval = int(current_interval - self.reward_interval + 1) # 5(time interval) - 1(interval/min)*5(min) + 1 = 1 
        queue_count = np.zeros((len(self.ramp_q)),dtype="float32")
        j = 0
        for num in self.ramp_q:
            count = 0
            q = self.Vissim.Net.QueueCounters.ItemByKey(num)
            for i in range(past_interval, current_interval + 1):
                temp = q.AttValue('QStops(Current, %i)'%i) or 0
                count += temp
            queue_count[j] = count/5
            j += 1
        return queue_count

    def get_tunnel_throughput(self):
        # 1800(s)//30(collect time interval) - (1800 - 600)//30 = 20
        current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - (1800 - 10*60)//30
        past_interval = int(current_interval - self.reward_interval*2 + 1) # 20(time interval) - 2(interval/min)*5(min) + 1 = 1 
        tunnel_throughput = np.zeros((len(self.tunnel_mileage_p)),dtype="float32")
        j = 0
        # run every vd
        for k,v in self.tunnel_mileage_p.items():
            f = 0
            dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range(past_interval, current_interval + 1):
                # check every feature is None type or not, for flow and occ are set None to 0, and for speed is set to 90(km/hr)
                temp = dc.AttValue("Vehs(Current,%i,All)"%i) or 0
                f += temp

            # save throughput every 5 min
            tunnel_throughput[j] = f
            j += 1
        return tunnel_throughput

    def get_merging_dif(self):
        # 1800(s)//30(collect time interval) - (1800 - 300)//30 = 10
        current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - int(1800 - 10*60 )//30
        past_interval = int(current_interval - self.reward_interval*2 + 1) # 10(time interval) - 2(interval/min)*5(min) + 1 = 1 
        merging_throughput = np.zeros((len(self.merging_mileage_p)),dtype="float32")
        j = 0
        # run every vd
        for k,v in self.merging_mileage_p.items():
            f = 0
            dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range(past_interval, current_interval + 1):
                # check every feature is None type or not, for flow and occ are set None to 0, and for speed is set to 90(km/hr)
                temp1 = dc.AttValue("Vehs(Current,%i,30)"%i) or 0
                temp2 = dc.AttValue("Vehs(Current,%i,10)"%i) or 0
                temp = temp1 - temp2
                f += temp

            # save mileage production every 5 min
            merging_throughput[j] = f
            j += 1
        return np.sum(merging_throughput)
    
    def get_metering_rate(self,kind):
        current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - (1800 - 10*60)//30
        past_interval = int(current_interval - self.reward_interval*2 + 1) # 20(time interval) - 2(interval/min)*5(min) + 1 = 1
        metering_rate = {}
        
        # run every vd
        for k,v in self.metering_rate_collector.items():
            f = 0
            mr = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range(past_interval, current_interval + 1):
                temp = mr.AttValue("Vehs(Current,%i,All)"%i) or 0
                f += temp

            metering_rate[k] = f
        if kind == 'm':
            return metering_rate['ramp']
        if kind == 'r':
            return metering_rate['mainline']

    # 以路網總車輛數取代旅行時間的計算
    def get_vehicles_num(self):# from '27.779' to '26.705'
        # 1800(s)//30(collect time interval) - (1800 - 600)//30 = 20
        current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - (1800 - 10*60)//30
        past_interval = int(current_interval - self.reward_interval*2 + 1) # 20(time interval) - 2(interval/min)*5(min) + 1 = 1 
        out = {}
        j = 0
        temp_dict = {'27.779':19,'26.705':17}
        # run every vd
        for k,v in temp_dict.items():
            f = 0
            dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range(past_interval, current_interval + 1):
                # check every feature is None type or not, for flow and occ are set None to 0, and for speed is set to 90(km/hr)
                temp = dc.AttValue("Vehs(Current,%i,All)"%i) or 0
                f += temp

            out[k] = f
            j += 1
        #print(out['27.779'] - out['26.705'])
        return out['27.779'] - out['26.705']

    def get_merging_num(self):# from '29.600' to '29.000'
        # 1800(s)//30(collect time interval) - (1800 - 600)//30 = 20
        current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - (1800 - 10*60)//30
        past_interval = int(current_interval - self.reward_interval*2 + 1) # 20(time interval) - 2(interval/min)*5(min) + 1 = 1
        out = {}
        j = 0
        #temp_dict = {'29.600':44,'29.000':21}
        temp_dict = {'B29.600':34,'29.000':21,'C29.600':22}
        # run every vd
        for k,v in temp_dict.items():
            f = 0
            dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range(past_interval, current_interval + 1):
                # check every feature is None type or not, for flow and occ are set None to 0, and for speed is set to 90(km/hr)
                temp = dc.AttValue("Vehs(Current,%i,All)"%i) or 0
                f += temp

            out[k] = f
            j += 1
        #print(out['29.600'] - out['29.000'])
        #return out['29.600'] - out['29.000']
        return out['B29.600']+out['C29.600'] - out['29.000']

    def get_occupancy(self):
        current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - (1800 - 10*60)//30
        past_interval = int(current_interval - self.reward_interval*2 + 1) # 20(time interval) - 2(interval/min)*5(min) + 1 = 1
        occupancy = {}
        
        # run every vd
        for k,v in self.bottleneck_mileage_p.items():
            o = 0
            mr = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range(past_interval, current_interval + 1):
                temp = mr.AttValue("OccupRate(Current,%i,All)"%i) or 0
                o += temp

            occupancy[k] = o/(current_interval-past_interval+1)
        
        # run every vd
        for k,v in self.merging_mileage_p.items():
            o = 0
            mr = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range(past_interval, current_interval + 1):
                temp = mr.AttValue("OccupRate(Current,%i,All)"%i) or 0
                o += temp

            occupancy[k] = o/(current_interval-past_interval+1)
        return occupancy
    
    def get_speed(self):
        current_interval = (int(self.Vissim.Simulation.AttValue('SimSec')//30) or int(self.simulation_time//self.control_cycle)) - (1800 - 10*60)//30
        past_interval = int(current_interval - self.reward_interval*2 + 1) # 20(time interval) - 2(interval/min)*5(min) + 1 = 1
        speed = {}
        
        # run every vd
        for k,v in self.bottleneck_mileage_p.items():
            o = 0
            mr = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range(past_interval, current_interval + 1):
                temp = mr.AttValue("SpeedAvgHarm(Current,%i,All)"%i) or 0
                o += temp

            speed[k] = o/(current_interval-past_interval+1)
        
        # run every vd
        for k,v in self.merging_mileage_p.items():
            o = 0
            mr = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range(past_interval, current_interval + 1):
                temp = mr.AttValue("SpeedAvgHarm(Current,%i,All)"%i) or 0
                o += temp

            speed[k] = o/(current_interval-past_interval+1)
        return speed
    
    # performance
    def export_bottleneck_throughput(self):
        bottleneck_throughput = {}
        # run every vd
        for k,v in self.bottleneck_mileage_p.items():
            f = 0
            dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range((self.warmup_time-1200)//self.control_cycle + 1, (self.simulation_time-1200)//self.control_cycle + 1):
                temp = dc.AttValue("Vehs(Current,%i,All)"%i) or 0
                f += temp

            # save mileage production every 5 min
            bottleneck_throughput[k] = f
        return bottleneck_throughput

    def export_merging_throughput(self): # merging throughput is only collected for buses.
        mileage_production = np.zeros((len(self.merging_mileage_p)),dtype="float32")
        j = 0
        # run every vd
        for k,v in self.merging_mileage_p.items():
            f = 0
            dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range((self.warmup_time-1200)//self.control_cycle + 1, (self.simulation_time-1200)//self.control_cycle + 1):
                temp = dc.AttValue("Vehs(Current,%i,30)"%i) or 0
                f += temp

            # save mileage production every 5 min
            mileage_production[j] = f
            j += 1
        return np.sum(mileage_production)

    def export_avg_travel_time(self):
        TTM = self.Vissim.Net.VehicleTravelTimeMeasurements
        travel_time = {}
        travel_time['Yilan-Tunnel Entry (Bus)'] = TTM.ItemByKey(self.tt_number['Yilan-Toucheng (Bus)']).AttValue('TravTm(Current,Avg,30)')\
                                                + TTM.ItemByKey(self.tt_number['Toucheng-Tunnel Entry (Bus)']).AttValue('TravTm(Current,Avg,30)')
        travel_time['Yilan-Tunnel_entry (Car)'] = TTM.ItemByKey(self.tt_number['Yilan-Toucheng (Car)']).AttValue('TravTm(Current,Avg,10)')\
                                                + TTM.ItemByKey(self.tt_number['Toucheng-Tunnel Entry (Car)']).AttValue('TravTm(Current,Avg,10)')
        travel_time['Entry-Exit (All)'] = TTM.ItemByKey(self.tt_number['Tunnel Entry-Exit']).AttValue('TravTm(Current,Avg,All)')
        
        travel_time['difference'] = travel_time['Yilan-Tunnel Entry (Bus)']\
                                  + TTM.ItemByKey(self.tt_number['Tunnel Entry-Exit']).AttValue('TravTm(Current,Avg,30)')\
                                  - travel_time['Yilan2Tunnel_entry (Car)']\
                                  - TTM.ItemByKey(self.tt_number['Tunnel Entry-Exit']).AttValue('TravTm(Current,Avg,10)')
        return travel_time

    def export_congestion_travel_time(self):
        TTM = self.Vissim.Net.VehicleTravelTimeMeasurements
        travel_time = {}
        travel_time['Yilan-Tunnel Entry (Bus)'] = np.nanmean(np.array([TTM.ItemByKey(self.tt_number['Yilan-Toucheng (Bus)']).AttValue('TravTm(Current,%i,30)'%i) for i in range(9,13)]))\
                                                + np.nanmean(np.array([TTM.ItemByKey(self.tt_number['Toucheng-Tunnel Entry (Bus)']).AttValue('TravTm(Current,%i,30)'%i) for i in range(9,13)]))
        travel_time['Yilan-Tunnel_entry (Car)'] = np.nanmean(np.array([TTM.ItemByKey(self.tt_number['Yilan-Toucheng (Car)']).AttValue('TravTm(Current,%i,10)'%i) for i in range(9,13)]))\
                                                + np.nanmean(np.array([TTM.ItemByKey(self.tt_number['Toucheng-Tunnel Entry (Car)']).AttValue('TravTm(Current,%i,10)'%i) for i in range(9,13)]))
        travel_time['Entry-Exit (All)'] = np.nanmean(np.array([TTM.ItemByKey(self.tt_number['Tunnel Entry-Exit']).AttValue('TravTm(Current,%i,All)'%i)for i in range(9,13)]))
        
        travel_time['difference'] = travel_time['Yilan-Tunnel Entry (Bus)']\
                                  + np.nanmean(np.array([TTM.ItemByKey(self.tt_number['Tunnel Entry-Exit']).AttValue('TravTm(Current,%i,30)'%i) for i in range(9,13)]))\
                                  - travel_time['Yilan-Tunnel_entry (Car)']\
                                  - np.nanmean(np.array([TTM.ItemByKey(self.tt_number['Tunnel Entry-Exit']).AttValue('TravTm(Current,%i,10)'%i) for i in range(9,13)]))
        return travel_time

    def export_travel_time(self):
        out = pd.DataFrame()
        TTM = self.Vissim.Net.VehicleTravelTimeMeasurements
        temp = pd.DataFrame()
        for k in self.tt_number:
            TTM_ = TTM.ItemByKey(self.tt_number[k])
            travel_time = np.array([TTM_.AttValue('TravTm(Current,%i,%s)'%(t,car)) for car in [30,10,'All'] for t in range(1,13)])
            section = np.array(np.repeat([k+'(Bus)', k+'(Car)', k+'(All)'],[12,12,12]),dtype=object)
            time = np.tile(np.arange(1,13),3)
            temp['Travel Time'] = travel_time
            temp['Section'] = section
            temp['Time'] = time
            out = pd.concat([out,temp],axis=0,ignore_index=True)
        return out

    def export_travel_flow(self):
        out = pd.DataFrame()
        TTM = self.Vissim.Net.VehicleTravelTimeMeasurements
        temp = pd.DataFrame()
        for k in self.tt_number:
            TTM_ = TTM.ItemByKey(self.tt_number[k])
            travel_flow = np.array([TTM_.AttValue('Vehs(Current,%i,%s)'%(t,car)) for car in [30,10,'All'] for t in range(1,13)])
            section = np.array(np.repeat([k+'(Bus)', k+'(Car)', k+'(All)'],[12,12,12]),dtype=object)
            time = np.tile(np.arange(1,13),3)
            temp['Travel Time'] = travel_flow
            temp['Section'] = section
            temp['Time'] = time
            out = pd.concat([out,temp],axis=0,ignore_index=True)
        return out

    # 檢驗合理，目前計算之平均為每30秒之排隊長度
    def export_mainline_queue(self):
        #self.mainline_q = {'0.5':23,'0.451':24,'0.419':25,'0.57':26,'0.58':27,'0.623':28,'0.907':29,'1.248':30,'1.175':31,'1.152':32,'5.134':33}
        queue_count = np.zeros(((self.simulation_time-1200)//self.control_cycle-(self.warmup_time-1200)//self.control_cycle),dtype="float32")
        m=0
        for i in range((self.warmup_time-1200)//self.control_cycle + 1, (self.simulation_time-1200)//self.control_cycle + 1):
            speed = np.zeros((len(self.mainline_q)),dtype="float32")
            occ   = np.zeros((len(self.mainline_q)),dtype="float32")
            j = 0
            # run every vd
            for k,v in self.mainline_q.items():
                s,o = 0,0
                dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
                # check every feature is None type or not, for flow and occ are set None to 0, and for speed is set to 90(km/hr)
                temp = dc.AttValue("SpeedAvgHarm(Current,%i,All)"%i) or 90
                s = temp

                temp = dc.AttValue("OccupRate(Current,%i,All)"%i) or 0
                o = temp

                speed[j] = s
                occ[j] = o
                j += 1
            #print(speed)
            #print(occ)

            for j,k,l in zip(speed,occ,self.mainline_q):       
                if j <= 40:
                    #print('flow %f speed %f occ %f'%(l,i,j))
                    queue_count[m]+=k*100*float(l)*2/0.965

            m+=1
        return queue_count

    def export_avg_ramp_queue(self):
        q = self.Vissim.Net.QueueCounters
        ramp_queue = []
        maxq = []
        for num in self.ramp_q:
            ramp_queue.append(q.ItemByKey(num).AttValue('QStops(Current, Avg)'))
            maxq.append(q.ItemByKey(num).AttValue('QStops(Current, Max)'))
        return np.sum(ramp_queue)

    def export_ramp_queue(self):
        q = self.Vissim.Net.QueueCounters
        ramp_queue = []
        for idx in range((self.warmup_time-1200)//self.control_cycle + 1, (self.simulation_time-1200)//self.control_cycle + 1):
            temp=0
            for num in self.ramp_q:
                temp+=q.ItemByKey(num).AttValue('QStops(Current, %i)'%idx) or 0
            ramp_queue.append(temp/2)
        return ramp_queue

    def export_travel_time_table(self,s=1,e=1):
        out = pd.DataFrame()
        for k in self.tt_number:
            out[k+'(Bus)'] = [0 for t in range(1,13)]
            out[k+'(Car)'] = [0 for t in range(1,13)]
            out[k+'(All)'] = [0 for t in range(1,13)]
        TTM = self.Vissim.Net.VehicleTravelTimeMeasurements
        for itr in range(s,e+1):
            temp = pd.DataFrame()
            for k in self.tt_number:
                TTM_ = TTM.ItemByKey(self.tt_number[k])
                temp[k+'(Bus)'] = [TTM_.AttValue('TravTm(%i,%i,30)'%(itr,t)) for t in range(1,13)]
                temp[k+'(Car)'] = [TTM_.AttValue('TravTm(%i,%i,10)'%(itr,t)) for t in range(1,13)]
                temp[k+'(All)'] = [TTM_.AttValue('TravTm(%i,%i,All)'%(itr,t)) for t in range(1,13)]
            out = out + temp
        out = out/len(range(s,e+1))
        return out

    def export_metering_rate(self):
        metering_rate = {}
        # run every vd
        for k,v in self.metering_rate_collector.items():
            f = 0
            mr = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range((self.warmup_time-1200)//self.control_cycle + 1, (self.simulation_time-1200)//self.control_cycle + 1):
                temp = mr.AttValue("Vehs(Current,%i,All)"%i) or 0
                f += temp

            # save mileage production every 5 min
            metering_rate[k] = f
        return metering_rate

    def export_tunnel_throughput(self):
        tunnel_throughput = {}
        # run every vd
        for k,v in self.tunnel_mileage_p.items():
            f = 0
            dc = self.Vissim.Net.DataCollectionMeasurements.ItemByKey(v)
            # run every time interval
            for i in range((self.warmup_time-1200)//self.control_cycle + 1, (self.simulation_time-1200)//self.control_cycle + 1):
                temp = dc.AttValue("Vehs(Current,%i,All)"%i) or 0
                f += temp

            tunnel_throughput[k] = f
        return tunnel_throughput

    def export_travel_time_delay(self):
        ttd = {'car':1,'bus':2}
        out = pd.DataFrame()
        TTD = self.Vissim.Net.DelayMeasurements
        temp = pd.DataFrame()
        TTD_ = TTD.ItemByKey(ttd['car'])
        travel_time_d1 = np.array([TTD_.AttValue('VehDelay(Current,%i,10)'%(t)) for t in range(1,13)])
        TTD_ = TTD.ItemByKey(ttd['bus'])
        travel_time_d2 = np.array([TTD_.AttValue('VehDelay(Current,%i,30)'%(t)) for t in range(1,13)])
        travel_time_d = np.concatenate((travel_time_d1,travel_time_d2),axis=0)    
        section = np.array(np.repeat(['car','bus'],[12,12]),dtype=object)
        time = np.tile(np.arange(1,13),2)
        temp['Travel Time Delay'] = travel_time_d
        temp['Type'] = section
        temp['Time'] = time
        out = pd.concat([out,temp],axis=0,ignore_index=True)
        return out